#!!pip install pyuff
import pyuff
import pprint

print ("Start read")
pp = pprint.PrettyPrinter(indent=4)



uff_file = pyuff.UFF('/Volumes/Harddriver/WTG01/209633-WTG01-2018-08-04-20-52-48_PwrAvg_543.uff')


print(len(uff_file.get_set_types()))

for i in range(len(uff_file.get_set_types())):
    print(i)
    print(" ")

    pp.pprint(uff_file.read_sets(i))
    print(" ")



