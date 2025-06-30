#関数で動作するようにします

import random
def omikuji():
    kuji = ["大吉", "中吉", "小吉", "末吉", "吉", "凶", "大凶"]
#    print(random.choice(kuji))
    return random.choice(kuji)
kekka = omikuji()
print("結果は" + kekka + "です")

