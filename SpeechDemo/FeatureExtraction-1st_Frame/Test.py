
best_loss=0
x=0
y=0
while True:
    try:
        c=x+y
        x=x+1
        print(x)
        if x > 3:
            print("through this--------")
            break
        continue
    except Exception as e:
        print("exception this-------")
        continue
    break