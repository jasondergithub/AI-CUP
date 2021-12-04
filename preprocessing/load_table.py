<<<<<<< HEAD
import pickle
import random

with open("../dict/relatedTable.txt", "rb") as fp:
    related_table = pickle.load(fp)

with open("../dict/unrelatedTable.txt", "rb") as fp:
    unrelated_table = pickle.load(fp)

name_list = ['table1.txt', 'table2.txt', 'table3.txt', 'table4.txt', 'table5.txt', 'table6.txt', 
'table7.txt', 'table8.txt', 'table9.txt', 'table10.txt']
for i in range(10):
    subtable = random.sample(unrelated_table, 1381)
    subtable = subtable + related_table
    random.shuffle(subtable)
    with open("../table/" + name_list[i], "wb") as fp:
=======
import pickle
import random

with open("../dict/relatedTable.txt", "rb") as fp:
    related_table = pickle.load(fp)

with open("../dict/unrelatedTable.txt", "rb") as fp:
    unrelated_table = pickle.load(fp)

name_list = ['table1.txt', 'table2.txt', 'table3.txt', 'table4.txt', 'table5.txt', 'table6.txt', 
'table7.txt', 'table8.txt', 'table9.txt', 'table10.txt']
for i in range(10):
    subtable = random.sample(unrelated_table, 1381)
    subtable = subtable + related_table
    with open("../table/" + name_list[i], "wb") as fp:
>>>>>>> c81d0f349c5860990a39bd8d48aba1fc5488af12
        pickle.dump(subtable, fp)