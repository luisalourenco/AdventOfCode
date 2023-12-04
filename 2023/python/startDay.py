import requests
import os
from datetime import date
import browser_cookie3
import sys

# 53616c7465645f5f5d61230bc5d6de273b2978806c2b7da95fe7f05b61c02ada6f9fe4958b9ffdaba3b5fadf9cf7c695e4c9aa59cdf9e1c1e214f9b1239e1aa7
#Get cookies from the browser
#cj = browser_cookie3.firefox()
cj = None
if not (".adventofcode.com" in str(cj)):
    cj = browser_cookie3.chrome()

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

#if not os.path.exists(f"day{day}"):
    #os.mkdir(f"day{day}")
os.chdir(f"../input")

headers = {
    'User-Agent': 'https://github.com/luisalourenco',
}


r = requests.get(f"https://adventofcode.com/2023/day/{day}/input", headers=headers, cookies = cj)
if len(str(day)) == 1:
    filename = f"day_0{day}.txt"
else:
    filename = f"day_{day}.txt"
    
with open(filename,"w") as f:
    f.write(r.text)