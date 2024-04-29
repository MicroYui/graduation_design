import random


# 随机生成一个长度为 n 的二进制串
def generate_binary_string(n):
    return ''.join(random.choice('01') for _ in range(n))


# 将二进制串解码为对应的十进制数
def decode_binary(binary_string):
    return int(binary_string, 2)


# 将十进制数编码为固定长度的二进制串
def encode_binary(decimal_number, length):
    binary_string = bin(decimal_number)[2:]
    padding_length = max(0, length - len(binary_string))
    return '0' * padding_length + binary_string


def encode_decimal(decimal_number, length, scale=100):
    scaled_number = int(decimal_number * scale)
    return encode_binary(scaled_number, length)


def decode_decimal(encoded_binary, scale=100):
    decoded_number = decode_binary(encoded_binary)
    return decoded_number / scale


# 示例
decimal_value = 1.5
binary_length = 7

encoded_binary = encode_decimal(decimal_value, binary_length)
decoded_decimal = decode_decimal(encoded_binary)

if __name__ == '__main__':
    print("Decimal:", decimal_value)
    print(type(decimal_value))
    print("Encoded Binary:", encoded_binary)
    print(type(encoded_binary))
    print("Decoded Decimal:", decoded_decimal)
    print(type(decoded_decimal))
