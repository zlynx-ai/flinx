

from flax import nnx, serialization
from pathlib import Path
from orbax import checkpoint as ocp
import json
import jax, jax.numpy as jnp

from .utils import get_dtype
from .infer import LanguageModel
from .. import models


class Flinx(nnx.Module):

    @classmethod
    def load(cls, path: str | Path, dtype=None):
        if isinstance(path, str):
            path = Path(path)

        if not path.is_absolute():
            path = path.resolve()

        if cls is Flinx:
            with open(path / "config.json", "r") as config_file:
                config: dict = json.load(config_file)

            arch = config.get("architecture")
            probably_config_class = arch.split("LanguageModel")[0].strip() + "Config"
            arch = getattr(models, arch)

        else:
            arch = cls
            with open(path / "config.json", "r") as config_file:
                config = json.load(config_file)

            arch_name = config.get("architecture", None)
            if arch_name is None:
                arch_name = arch.__name__

            probably_config_class = (
                arch_name.split("LanguageModel")[0].strip() + "Config"
            )

        processor_class = getattr(arch, "processor", None)
        if processor_class is not None:
            processor = processor_class.load(path)

        config_class = getattr(models, probably_config_class, None)
        if probably_config_class is None:
            from .config import LanguageConfig

            config_class = LanguageConfig

        config = config_class(**config)

        model = nnx.eval_shape(lambda: arch(config=config))
        gdef, abs_state = nnx.split(model)

        mesh = jax.sharding.Mesh(jax.devices(), ("model",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("model"))

        def wrap_with_sharding(leaf):
            rank = len(leaf.shape)

            if rank == 0:
                spec = jax.sharding.PartitionSpec()
            else:
                spec = jax.sharding.PartitionSpec("model")

            actual_sharding = jax.sharding.NamedSharding(mesh, spec)

            return jax.ShapeDtypeStruct(
                shape=leaf.shape, dtype=leaf.dtype, sharding=actual_sharding
            )

        abs_state = jax.tree.map(wrap_with_sharding, abs_state)

        ckpter = ocp.StandardCheckpointer()
        state = ckpter.restore(path, abs_state)
        ckpter.wait_until_finished()

        if dtype is not None:
            target_dtype = get_dtype(dtype) if isinstance(dtype, str) else dtype
            def _cast(x):
                if hasattr(x, "dtype") and x.dtype != target_dtype and jnp.issubdtype(x.dtype, jnp.floating):
                    return x.astype(target_dtype)
                return x
            state = jax.tree.map(_cast, state)

        model = nnx.merge(gdef, state)

        return model, processor

    def save(self, path: str | Path):
        if isinstance(path, str):
            path = Path(path)

        if not path.is_absolute():
            path = path.resolve()

        checkpointer = ocp.StandardCheckpointer()

        _, state = nnx.split(self)
        try:
            arch = type(self)
            config = getattr(self, "config", self.kwargs.get("config", None))

            if config is None:
                raise Exception(
                    f"config is None type, to resolve this send config instance with super().__init__(config=config)."
                )

            checkpointer.save(path, state)
            checkpointer.wait_until_finished()

            with open(path / "config.json", "w") as config_file:
                json.dump(serialization.to_state_dict(config), config_file, indent=2)

            print(f"save model path {path}.")
        except Exception as e:
            print(e)



    def load_from_config(): 
        """
        handle `load` method if specify load from config
        """
        ...


    def load_from_ckpt(): 
        """
        handle `load` method if specify load from checkpoint
        """
        ...