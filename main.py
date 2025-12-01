import importlib
import sys
from pathlib import Path

from clay.layout import fit
from clay.render.matplot import render


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <module_name>")
        sys.exit(1)
    
    module_name = sys.argv[1]
    
    # Load module from examples package
    try:
        module = importlib.import_module(f"examples.{module_name}")
    except ModuleNotFoundError:
        raise Exception(f"Module 'examples.{module_name}' not found")
    
    # Check if module has a graph member
    if not hasattr(module, "graph"):
        raise Exception(f"Module 'examples.{module_name}' does not have a 'graph' member")
    
    g = module.graph
    result = fit(g)
    output_path = Path("output") / f"{module_name}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render(result.layout, str(output_path))
    print(f"Successfully rendered graph from examples.{module_name}")
    print(f"Optimization details: {result.optimization_result}")


if __name__ == "__main__":
    main()
