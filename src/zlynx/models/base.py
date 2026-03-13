

from flax import nnx, serialization, struct
from pathlib import Path
from orbax import checkpoint as ocp
from typing import Literal, List, Dict, Tuple, Set
from dataclasses import field
from safetensors.flax import save_file, load_file
import os
import json
import jax, jax.numpy as jnp


from ..utils import get_dtype
from .. import models

import logging
import shutil
import tempfile
from datetime import datetime

logging.basicConfig(level=logging.INFO, force=True)
logging.getLogger('absl').setLevel(logging.WARNING)
logging.getLogger('jax').setLevel(logging.WARNING)

class Z(nnx.Module):

    # return config
    @classmethod
    def load_config(cls, path: str | Path, asdict = False, config_map: Dict = {}):
        path = Path(path).resolve()

        assert (path / "config.json").exists(), \
            f"model config not found in dir {path}"

        with open(path / "config.json", "r") as config_file:
            config_dict: dict = json.load(config_file)

        config_dict = {config_map[k] if k in config_map else k:v for k, v in config_dict.items()}

        if asdict:
            return config_dict

        config_class = None
        config_class_name = config_dict.get("conf", None) 
        
        if config_class_name is not None:

            # try in-lib model config
            config_class = getattr(models, config_class_name, None)
            
            # try user defined
            if config_class is None:
                config_class = globals().get(config_class_name)

        # still None return as C base config
        if config_class is None:
            from .config import C
            config_class = type('Config', (C,), {
                '__annotations__': {**C.__annotations__, **{k: type(v) for k, v in config_dict.items()}},
                **{k: v if not isinstance(v, (List, Dict, Tuple, Set)) else field(default_factory=lambda v=v: v) for k, v in config_dict.items()}
            })
            config_class = struct.dataclass(config_class)

        return config_class(**config_dict)

    def _load_safetensors(state: nnx.State, path: str, module_map: Dict={}):

        index_path = os.path.join(path, "model.safetensors.index.json")
        with open(index_path, "r") as f:
            index_data = json.load(f)

        weight_map = index_data["weight_map"]

        files_to_load = {}
        for k_str, file_name in weight_map.items():
            if file_name not in files_to_load:
                files_to_load[file_name] = []
            files_to_load[file_name].append(k_str)

        current_state = nnx.to_flat_state(state)
        new_state = []

        for file_name, keys_in_file in files_to_load.items():
            shard_path = os.path.join(path, file_name)

            shard_data = load_file(shard_path) 
            
            for k_str in keys_in_file:
                k_tuple = tuple(
                    int(m) if m.isdigit() \
                        else module_map[m] \
                            if m in module_map \
                                else m for m in k_str.split(".")
                )
                
                if k_tuple in current_state._keys:
                    instance = type(dict(current_state)[k_tuple])
                    sharding = dict(current_state)[k_tuple].sharding
                    value = shard_data[k_str]
                    
                    new_state.append((k_tuple, jax.device_put(instance(value), sharding)))
                else:
                    print(f"{k_str} not found in the model")
            
            del shard_data

        nnx.update(state, nnx.FlatState(new_state, sort=False).to_nested_state())
        return state

    @classmethod
    def load(
        cls, path: str | Path, 
        *,
        dtype=None, 
        config=None, 
        config_map: Dict={},
        module_map: Dict={},
        sharding=None,
        format: Literal["orbax", "safetensors"]="orbax",
        **kwargs
    ):
        path = Path(path).resolve()

        if config is None:
            config = cls.load_config(path, config_map=config_map)

        if cls is Z:
            arch_name = None
            if config is not None:
                arch_name = getattr(config, "architecture", None) \
                    or getattr(config, "architectures", None) \
                    or getattr(config, "arch", None)

            if isinstance(arch_name, List):
                raise ValueError("Not support loading this model from Z class.")
                
            if arch_name is None:
                raise ValueError("Could not determine model architecture.")
            
            arch = getattr(models, arch_name, None)
        else:
            # allow config = None
            arch = cls
        
        logging.info(f"{arch.__name__} model class obtained")


        if config is None:
            model = nnx.eval_shape(lambda: arch(**kwargs))
        else:
            model = nnx.eval_shape(lambda: arch(config=config, **kwargs))
        
        gdef, state = nnx.split(model)

        if sharding is not None:

            # auto sharding
            if sharding == "ddp":
                mesh = jax.sharding.Mesh(jax.devices(), ("data",))
            elif sharding == "fsdp":
                mesh = jax.sharding.Mesh(jax.devices(), ("model",))

            def wrap_with_sharding(leaf):
                shape = leaf.shape
                rank = len(shape)
                shard_dim = shape[0] if rank > 0 else None

                if rank == 0 or sharding == "ddp":
                    spec = jax.sharding.PartitionSpec()
                elif shard_dim is not None and shard_dim % jax.device_count() != 0:
                    spec = jax.sharding.PartitionSpec()
                else:
                    spec = jax.sharding.PartitionSpec("model")

                actual_sharding = jax.sharding.NamedSharding(mesh, spec)

                return jax.ShapeDtypeStruct(
                    shape=leaf.shape, dtype=leaf.dtype, sharding=actual_sharding
                )

            state = jax.tree.map(wrap_with_sharding, state)

        if format == "safetensors":
            state = Z._load_safetensors(state, path, module_map)

        if format == "orbax":
            ckpter = ocp.StandardCheckpointer()
            state = ckpter.restore(path, state)
            ckpter.wait_until_finished()

        if dtype is not None:
            target_dtype = get_dtype(dtype) if isinstance(dtype, str) else dtype
            def _cast(x):
                if hasattr(x, "dtype") and x.dtype != target_dtype and jnp.issubdtype(x.dtype, jnp.floating):
                    return x.astype(target_dtype)
                return x
            state = jax.tree.map(_cast, state)

        model = nnx.merge(gdef, state)

        processor = None
        # if processor class is set in model class
        processor_class = getattr(arch, "processor", None)
        if processor_class is None:
            if config is not None:
                # maybe in the models lib
                processor_class_name = getattr(config, "processor", None)
                if processor_class_name is not None:
                    processor_class = getattr(models, processor_class_name, None)
                    if processor_class is None:
                        # maybe it's user defined
                        processor_class = getattr(globals(), processor_class_name, None)

        if processor_class is None:
            logging.warning("No processor in this model, return None")

        else:
            processor = processor_class.load(path)
            if processor is not None:
                logging.info("The model processor has been loaded")

        setattr(model, "processor", processor)
        return model, processor
    
    def _save_safetensors(self, path, max_shard_size_gb=3):
        
        os.makedirs(path, exist_ok=True)

        state = nnx.state(self)

        flat_state = nnx.to_flat_state(state=state)

        max_shard_size = int(max_shard_size_gb * 1024**3)
        
        shards = []
        current_shard = {}
        current_size = 0

        for k, v in dict(flat_state).items():
            k = ".".join(map(str, k))

            if not hasattr(v, "shape") or len(v.shape) == 0:
                v = jnp.array(v)
                
            v_size = v.nbytes

            if current_size + v_size > max_shard_size and current_shard:
                shards.append(current_shard)
                current_shard = {}
                current_size = 0
                
            current_shard[k] = v
            current_size += v_size

        if current_shard:
            shards.append(current_shard)

        num_shards = len(shards)
        weight_map = {}
        total_size = 0

        for i, shard in enumerate(shards, 1):
            file_name = f"model-{i:05d}-of-{num_shards:05d}.safetensors"
            save_path = os.path.join(path, file_name)
            
            save_file(shard, save_path)
            
            for k, v in shard.items():
                weight_map[k] = file_name
                total_size += v.nbytes

        index_data = {
            "metadata": {
                "total_size": total_size
            },
            "weight_map": weight_map
        }

        with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
            json.dump(index_data, f, indent=2)

    def save(
        self, path: str | Path, *, 
        format: Literal["orbax", "safetensors", "all"] = "orbax", 
        max_shard_size_gb=3
    ):
        if isinstance(path, str):
            path = Path(path)

        if not path.is_absolute():
            path = path.resolve()

        with tempfile.TemporaryDirectory(dir=path.parent, prefix="zlynx_ckpt_") as tmp_dir:
            tmp_path = Path(tmp_dir)

            try:
                
                config = getattr(self, "config", self.kwargs.get("config", None) if hasattr(self, "kwargs") else None)

                if tmp_path.exists():
                    shutil.rmtree(tmp_path)
                
                if format in ["orbax", "all"]:
                    state = nnx.state(self)
                    checkpointer = ocp.StandardCheckpointer()
                    checkpointer.save(tmp_path, state)
                    checkpointer.wait_until_finished()
                
                if format in ["safetensors", "all"]:
                    self._save_safetensors(tmp_path, max_shard_size_gb=max_shard_size_gb)

                processor = getattr(self, "processor", None)
                if processor is not None:
                    processor.save(tmp_path)

                if config is not None:
                    with open(tmp_path / "config.json", "w") as config_file:
                        json.dump(serialization.to_state_dict(config), config_file, indent=2)

                if jax.process_index() == 0:
                    if path.exists():
                        shutil.rmtree(path)

                    tmp_path.rename(path)

                    logging.info(f"Save model path {path}.")

            except Exception as e:
                logging.error(e)


    def push_hf(
        self, repo_id, 
        private,
        *,
        format: Literal["orbax", "safetensors"]="safetensors", 
        max_shard_size_gb: float = 3,
        **kwargs
    ):
        from huggingface_hub import create_repo, upload_folder

        repo_type = "model"

        repo = create_repo(
            repo_id=repo_id, 
            private=private,
            repo_type=repo_type,
            **kwargs
        )
        repo_id = repo.repo_id


        with tempfile.TemporaryDirectory(dir=".", prefix="zlynx_ckpt_") as tmp_dir:
            tmp_path = Path(tmp_dir)

            try:
                
                config = getattr(self, "config", self.kwargs.get("config", None) if hasattr(self, "kwargs") else None)

                if tmp_path.exists():
                    shutil.rmtree(tmp_path)
                
                if format in ["orbax", "all"]:
                    state = nnx.state(self)
                    checkpointer = ocp.StandardCheckpointer()
                    checkpointer.save(tmp_path, state)
                    checkpointer.wait_until_finished()
                
                if format in ["safetensors", "all"]:
                    self._save_safetensors(tmp_path, max_shard_size_gb=max_shard_size_gb)

                processor = getattr(self, "processor", None)
                if processor is not None:
                    processor.save(tmp_path)

                if config is not None:
                    with open(tmp_path / "config.json", "w") as config_file:
                        json.dump(serialization.to_state_dict(config), config_file, indent=2)

                if jax.process_index() == 0:
                    upload_folder(
                        folder_path=tmp_path,
                        repo_id=repo_id,
                        repo_type=repo_type,
                        **kwargs
                    )

                    logging.info(f"Pushed model to HuggingFace https://huggingface.co/{repo_id}")

            except Exception as e:
                logging.error(e)

    def push_kaggle(
        self, repo_id, 
        variation="default", *, 
        format: Literal["orbax", "safetensors"]="safetensors", 
        max_shard_size_gb: int=3, **kwargs
    ):
        """
        ### login kaggle:
        ```python
        import kagglehub
        kagglehub.login()
        ```
        """
        import kagglehub

        date = datetime.now().strftime("%Y-%m-%d")

        with tempfile.TemporaryDirectory(dir=".", prefix="zlynx_ckpt_") as tmp_dir:
            tmp_path = Path(tmp_dir)

            try:
                
                config = getattr(self, "config", self.kwargs.get("config", None) if hasattr(self, "kwargs") else None)

                if tmp_path.exists():
                    shutil.rmtree(tmp_path)
                
                if format in ["orbax", "all"]:
                    state = nnx.state(self)
                    checkpointer = ocp.StandardCheckpointer()
                    checkpointer.save(tmp_path, state)
                    checkpointer.wait_until_finished()
                
                if format in ["safetensors", "all"]:
                    self._save_safetensors(tmp_path, max_shard_size_gb=max_shard_size_gb)

                processor = getattr(self, "processor", None)
                if processor is not None:
                    processor.save(tmp_path)

                if config is not None:
                    with open(tmp_path / "config.json", "w") as config_file:
                        json.dump(serialization.to_state_dict(config), config_file, indent=2)

                if jax.process_index() == 0:
                    kagglehub.model_upload(
                        handle = f"{repo_id}/flax/{variation}",
                        local_model_dir = str(tmp_path),
                        version_notes = f'Update {date}',
                        **kwargs
                    )

                    logging.info(f"Pushed model to Kaggle https://www.kaggle.com/models/{repo_id}")

            except Exception as e:
                logging.error(e)

    @classmethod
    def load_hf(
        cls, repo_id: str, 
        *,
        dtype=None, 
        config=None, 
        config_map: Dict={},
        module_map: Dict={},
        sharding=None,
        format: Literal["orbax", "safetensors"]="safetensors",
        hf_kwargs: dict={},
        **model_kwargs
    ):
        
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(repo_id=repo_id, repo_type="model", **hf_kwargs)
        path = Path(local_dir).resolve()

        model, processor = cls.load(
            path, dtype=dtype, 
            config=config, 
            sharding=sharding, 
            format=format, 
            config_map=config_map,
            module_map=module_map,
            **model_kwargs
        )
        return model, processor
    

    @classmethod
    def load_kaggle(
        cls, repo_id: str, 
        variation: str="default",
        *,
        dtype=None, 
        config=None, 
        config_map: Dict={},
        module_map: Dict={},
        sharding=None,
        format: Literal["orbax", "safetensors"]="safetensors",
        kaggle_kwargs: dict={},
        **model_kwargs
    ):
        import kagglehub

        local_dir = kagglehub.model_download(f"{repo_id}/flax/{variation}", **kaggle_kwargs)
        path = Path(local_dir).resolve()

        model, processor = cls.load(
            path, dtype=dtype, 
            config=config, 
            sharding=sharding, 
            format=format, 
            config_map=config_map,
            module_map=module_map,
            **model_kwargs
        )
        return model, processor