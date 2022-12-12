def format_num(n):
    f = '{0:.3g}'.format(n)
    f = f.replace('+0', '+')
    f = f.replace('-0', '-')
    n = str(n)
    return f if len(f) < len(n) else n


f = 3.14159

print(format_num(f))