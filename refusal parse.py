import re

text = "Representative name: weporiweporipweoir \n wwewewqe"


pattern = r"Representative name: (.+)"
match = re.search(pattern, text)

if match:
    representative_name = match.group(1)
    print(representative_name)
else:
    print("Representative name not found.")