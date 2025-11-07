product_list_PD1 = []
product_list_PD2 = []
inertia_list_PD1 = inertia_values[0:3]
inertia_list_PD2 = inertia_values[3:len(inertia_values)]

for i in range(3):
    product_PD1 = math.prod(ratio_list[0:i])
    product_list_PD1.append(product_PD1)
    #print(math.prod(product_list_PD1[0:i]))

for i in range(3,len(ratio_list)):
    product_PD2 = math.prod(ratio_list[3:i])
    product_list_PD2.append(product_PD2)
    #print(math.prod(product_list_PD2[0:i]))
    
product_list_PD1_r = product_list_PD1[::-1] 
product_list_PD2.append(product_list_PD2[-1])

print("Product_List_PD2", product_list_PD2)

reduced_inertia_list_PD1 = [inertia_list_PD1[i] * product_list_PD1_r[i] for i in range(len(inertia_list_PD1))]  
reduced_inertia_list_PD2 = [inertia_list_PD2[i] * product_list_PD2[i] for i in range(len(inertia_list_PD2))]  
print("red. I PD1",reduced_inertia_list_PD1)
print("red. I PD2",reduced_inertia_list_PD2)



#for everything behing (-> Outshaft, Wheel,...) use the "normal" ratio list
print("orig. ratios",ratio_list)
print("Orig. inertias",inertia_values)
