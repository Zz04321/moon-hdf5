# Optional pre-commit check

The hook runs `moon check`, independently of GitHub Actions.
Enable it from the repository root with MoonBit on `PATH`:

```sh
chmod +x .githooks/pre-commit
git config core.hooksPath .githooks
```
