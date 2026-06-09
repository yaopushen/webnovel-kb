"""中文数字转换工具。"""


def int_to_cn(num: int) -> str:
    """将正整数转换为中文数字。例如: 1→一, 12→十二, 100→一百, 10000→一万。"""
    if num <= 0:
        return str(num)
    cn_digits = "零一二三四五六七八九"
    if num < 10:
        return cn_digits[num]
    parts = []
    if num >= 10000:
        w = num // 10000
        parts.append(f"{int_to_cn(w)}万")
        num %= 10000
    if num >= 1000:
        q = num // 1000
        parts.append(f"{cn_digits[q]}千")
        num %= 1000
    if num >= 100:
        b = num // 100
        parts.append(f"{cn_digits[b]}百")
        num %= 100
    if num >= 10:
        s = num // 10
        if s > 1:
            parts.append(f"{cn_digits[s]}十")
        elif parts:  # s==1, 有更高位: 110→一百一十
            parts.append("一十")
        else:  # s==1, 无更高位: 10→十
            parts.append("十")
        num %= 10
    if num > 0:
        parts.append(cn_digits[num])
    return "".join(parts)
