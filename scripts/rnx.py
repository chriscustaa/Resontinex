# scripts/rnx.py
import typer
import json
from resontinex.config import RuntimeConfig

app = typer.Typer()

@app.command()
def sample():
    """Prints a sample JSON configuration."""
    print(json.dumps(RuntimeConfig().dict(), indent=2))

@app.command()
def validate(path: str):
    """Validates a JSON configuration file."""
    try:
        RuntimeConfig.parse_raw(open(path, "r", encoding="utf-8").read())
        typer.echo("OK")
    except Exception as e:
        typer.echo(f"Validation Error: {e}", err=True)
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()