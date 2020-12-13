import requests
import os
from datetime import date
import browser_cookie3
import sys

#Get cookies from the browser
try:
    cj = browser_cookie3.firefox()
    if not (".adventofcode.com" in str(cj)):
        cj = browser_cookie3.chrome()
except:
    cj = '53616c7465645f5f7c0e1ad1e462e8e5b89b93abc5e32fa2db42ccbda7c6a44161e5eecd9d18a385bab0edd0f0bb1a6a'

#Get today number of day
day_today = date.today().strftime("%d").lstrip("0")

#If we provide an argument, use it as the desired day. Ex: ./startDay.py 5. Otherwise use day_today
if len(sys.argv) > 1:
    day = int(sys.argv[1])
    if day<0 or day>31 or day>int(day_today):
        exit("Day is not valid")
else:
    day = day_today


print(f"Initializing day {day}")

if not os.path.exists(f"day{day}"):
    os.mkdir(f"day{day}")
    os.chdir(f"day{day}")
    r = requests.get(f"https://adventofcode.com/2019/day/{day}/input", cookies = cj)
    with open(f"day{day}","w") as f:
        f.write(r.text)