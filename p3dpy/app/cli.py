from .vizserver import create_app, set_parameters


app = create_app()

def main():
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Visualization server for p3dpy.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address.")
    parser.add_argument("--port", type=int, default=8000, help="Port number.")
    parser.add_argument("--params", type=str, default="{}", help="Parameters on JSON format.")
    args = parser.parse_args()
    set_parameters(args.params)
    uvicorn.run(app=app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
