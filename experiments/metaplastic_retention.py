from ._common import run_named

if __name__ == "__main__":
    run_named("configs/metaplastic.yaml", "metaplastic_retention", include_bypass=False)
