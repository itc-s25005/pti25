# エントリーポイントを設定したbmi.py

def main():
    h = float(input("身長何cmですか？")) / 100.0
    w = float(input("体重何kgですか？"))
    bmi = w / (h * h)
    print("あなたのBMI値は、",bmi,"です。")

if __name__ == "__main__":
    main()
