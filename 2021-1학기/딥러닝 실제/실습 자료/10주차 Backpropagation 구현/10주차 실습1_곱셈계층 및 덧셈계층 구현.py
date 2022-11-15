class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
        dx = dout * self.y  # x와 y를 바꾼다.
        dy = dout * self.x

        return dx, dy

class AddLayer:
    def __init__(self): # 초기화 필요x (곱셈에서는 상대방의 값을 곱해야해서 변수들의 값 저장했지만, 얘는 필요x )
        pass

    def forward(self, x, y):
        out = x + y

        return out

    def backward(self, dout): # 입력받은 신호 그대로 출력
        dx = dout * 1
        dy = dout * 1

        return dx, dy

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# layer 인스탄스 4개 생성.
mul_apple_layer = MulLayer()        # 노드 1 (필기에 표시한거)
mul_orange_layer = MulLayer()       # 노드 2
add_apple_orange_layer = AddLayer() # 노드 3
mul_tax_layer = MulLayer()          # 노드 4

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)                # (1)
orange_price = mul_orange_layer.forward(orange, orange_num)            # (2)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)  # (3)
price = mul_tax_layer.forward(all_price, tax)                          # (4)

# backward
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)                          # (4)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)  # (3)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)            # (2)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)                # (1)

print("price:", int(price))
print("dApple:", dapple)
print("dApple_num:", int(dapple_num))
print("dOrange:", dorange)
print("dOrange_num:", int(dorange_num))
print("dTax:", dtax)