import psutil

# Function to kill the process using a specific port
def kill_process_on_port(port):
    for conn in psutil.net_connections(kind='inet'):
        if conn.laddr.port == port:
            try:
                # Make sure the process ID is valid and accessible
                proc = psutil.Process(conn.pid)
                print(f"Terminating process {proc.pid} using port {port}")
                proc.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                # Handle errors if process is not accessible or doesn't exist
                print(f"Unable to terminate process with pid {conn.pid}, possibly due to permissions or it being a system process.")

# Kill processes using the ports 8100, 9100, and 3000 before starting the servers
kill_process_on_port(8100)
kill_process_on_port(9100)
kill_process_on_port(3000)
