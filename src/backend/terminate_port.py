import psutil
import os

def kill_process_on_port(port):
    current_pid = os.getpid()  # PID för processen som kör detta script

    for conn in psutil.net_connections(kind='inet'):
        if conn.laddr.port == port and conn.pid and conn.pid != current_pid:
            try:
                proc = psutil.Process(conn.pid)
                print(f"Terminating process {proc.pid} using port {port} ({proc.name()})")
                proc.terminate()
                proc.wait(timeout=3)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
                print(f"⚠️ Could not terminate process with pid {conn.pid}: {e}")
            except psutil.TimeoutExpired:
                print(f"⚠️ Process {conn.pid} did not terminate in time.")

# Kill processes using the ports 8100, 9100, and 3000 before starting the servers
kill_process_on_port(8100)
kill_process_on_port(9100)
kill_process_on_port(3000)
