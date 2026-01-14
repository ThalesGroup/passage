# PASSAGE Usage

Generate dataset for a single resolution (overrides `config.yaml`):

```bash
passage generate --resolution 256
```

Generate datasets for multiple resolutions at once:

```bash
passage generate --resolution 64 128 256
```

If `--resolution` is not provided, resolutions are taken from `config/config.yaml` (the `resolutions` key).
