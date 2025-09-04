import subprocess, platform


def history_clear():
    subprocess.run("history -c; history -w", shell=True)
    subprocess.run("sudo rm ~/.bash_history", shell=True)


def journal_clear():
    subprocess.run("sudo journalctl --vacuum-time=1s", shell=True)
    subprocess.run("sudo journalctl --vacuum-size=1M", shell=True)
    subprocess.run("sudo find /var/log/journal -name '*.journal' | xargs sudo rm", shell=True)
    subprocess.run("sudo rm -rf /var/log/journal/*", shell=True)
    subprocess.run("sudo mkdir -p /var/log/journal", shell=True)
    subprocess.run("sudo systemctl restart systemd-journald", shell=True)


def logfile_write_zero():
    subprocess.run(r"sudo find /var/log/ -type f -name \* -exec cp -f /dev/null {} \;", shell=True)


def package_cache_clear():
    subprocess.run("sudo apt-get clean", shell=True)
    subprocess.run("sudo apt-get autoclean", shell=True)
    subprocess.run("sudo rm -rf /var/lib/apt/lists/*", shell=True)
    subprocess.run("sudo rm -rf /var/cache/apt/", shell=True)


def cache_clear():
    subprocess.run("sudo rm -rf ~/.cache/*", shell=True)
    subprocess.run("sudo rm -rf /var/cache/*", shell=True)
    subprocess.run("sudo rm -rf /tmp/*", shell=True)


def browser_cache_clear():
    subprocess.run("sudo rm -rf ~/.mozilla/firefox/*/cache2/*", shell=True)
    subprocess.run("sudo rm -rf ~/.cache/google-chrome/*", shell=True)
    subprocess.run("sudo rm -rf ~/.cache/chromium/*", shell=True)


def log_clear():
    if platform.system() == "Linux":
        package_cache_clear()
        cache_clear()
        browser_cache_clear()
        history_clear()
        journal_clear()
        logfile_write_zero()
        print("Log and cache cleared successfully.")
    else:
        print("This script only supports Linux system")


log_clear()
