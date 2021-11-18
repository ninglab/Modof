import pickle
import pdb

def save(data, path, idx1, idx2):
    file_path = open(path, 'w')
    for tmp in data:
        if tmp[idx1] < tmp[idx2]:
            file_path.write("%s %s %.4f %.4f %.4f\n" % (tmp[0], tmp[1], tmp[2], tmp[idx1], tmp[idx2]))
        else:
            file_path.write("%s %s %.4f %.4f %.4f\n" % (tmp[1], tmp[0], tmp[2], tmp[idx2], tmp[idx1]))
    file_path.close()

data = pickle.load(open("./pair_with_prop.pkl", 'rb'))

qed_test = [line.strip() for line in open("../data/qed/test.txt", 'r').readlines()]
qed_1 = [tmp for tmp in data if min(tmp[6], tmp[7]) < 0.6 \
                            and max(tmp[6], tmp[7]) >= 0.7 \
                            and tmp[0] not in qed_test]

save(qed_1, "../data/qed_1/pairs.txt", 6, 7)

#drd2_test = [line.strip() for line in open("../data/drd2/test.txt", 'r').readlines()]
#drd2_2 = [tmp for tmp in data if abs(tmp[9] - tmp[10]) >= 0.2 \
#                                 and min(tmp[9], tmp[10]) < 0.5 \
#                                 and tmp[0] not in drd2_test \
#                                 and tmp[1] not in drd2_test]

#save(drd2_2, "../data/drd2_25/pairs.txt", 9, 10)


