import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
import mlxtend.frequent_patterns as mlx
from collections import Counter

def loadDataSet():
    test_data = "avengers_data.txt"
    data_file = open(test_data)
    lines = data_file.readlines()
    dataset = []
    customer_info = []
    for line in lines:
        separated = line.replace("\n", "").split(" ")
        customer_info.append(separated[:2])
        dataset.append(separated[2:])
    return (dataset, customer_info)

def main():
    transcation_table, customer_info = loadDataSet()
    encoder = TransactionEncoder()
    encoder_ary = encoder.fit(transcation_table).transform(transcation_table)
    df = pd.DataFrame(encoder_ary, columns=encoder.columns_)
    #print(df)
    frequent_sets = mlx.apriori(df, min_support=0.5, use_colnames=True)
    rules = mlx.association_rules(frequent_sets, metric="confidence", min_threshold=0.7) # pandas DataFrame

    purchase_history = input("What have you purchased? ").replace(" ", "").lower().split(",")
    lift_values = []
    consequents = []
    print(rules)
    for i in range(len(rules.values)):
        if Counter(purchase_history) == Counter(rules.values[i][0]):
            lift_values.append(rules.values[i][len(rules.columns)-3])
            consequents.append(rules.values[i][1])

    if len(lift_values) == 0:
        print("\nNo predicted purhcases.")
        return

    prediction = consequents[lift_values.index(max(lift_values))]
    print("\nYour predicted future purchases:")
    predicted_names = ", ".join(prediction)
    print(predicted_names)

if __name__ == '__main__':
    main()
