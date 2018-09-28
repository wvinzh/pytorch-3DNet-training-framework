#-*- coding: UTF-8 -*- 
import asyncio
import time
start_time = time.time()

async def compute(x, y):
    print("Compute %s + %s ..." % (x, y))
    await asyncio.sleep(0)
    return x + y

async def print_sum(x, y):
    result = await compute(x, y)
    print("%s + %s = %s" % (x, y, result))

loop = asyncio.get_event_loop()
loop.run_until_complete(asyncio.gather(print_sum(1, 2),print_sum(3, 4),print_sum(5, 6),print_sum(7, 8),print_sum(9, 10),print_sum(11, 12)))
loop.close()

'''
def compute(x, y):
    print("Compute %s + %s ..." % (x, y))
    time.sleep(1)
    return x + y

def print_sum(x,y):
  result = compute(x, y)
  print("%s + %s = %s" % (x, y, result))

if __name__=='__main__':
  print_sum(1, 2)
  print_sum(3, 4)
  print_sum(5, 6)
  print_sum(7, 8)
  print_sum(9, 10)
  print_sum(11, 12)'''


duration = time.time() - start_time
print('test time %.3f s' % duration)  
