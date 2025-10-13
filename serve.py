from livereload import Server


def main() -> None:
    server = Server()
    # Watch project files
    server.watch('index.html')
    server.watch('batch.html')
    server.watch('styles.css')
    server.watch('src/*.js')

    # Serve current directory on port 5173
    # livereload injects a small script into HTML responses to auto-refresh the page
    server.serve(port=5173, root='.')


if __name__ == '__main__':
    main()


