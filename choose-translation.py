import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

if __name__ == "__main__":
    f = open(sys.argv[1], 'r', encoding="utf-8")
    w = open(sys.argv[2], 'w')
    
    res = []
    for i in range(5000):
        res.append('')
    
    a = f.readline()
    
    while a:
        if a[0] == 'H':
            tmp = a.strip().split('\t')
            res[int(tmp[0][2:]) ] = tmp[-1]
            res[int(tmp[0][2:]) ] += '\n'
        a = f.readline()

    for item in res:
        w.write(item)

