import math
from Parse_Parameters import parse_body_values, parse_node_values

def reduce_to_slowest(inertia_dict, spring_data):

    # creating a list of all stiffness-values, e.g.
    stiffness_list = [props['stiffness'] for props in spring_data.values()]
    print("All stiffnesses:", stiffness_list)

    # creating a list of all stiffness-values, e.g.
    ratio_list = [props['ratio'] for props in spring_data.values()]
    print("All ratios:", ratio_list)

    # creating a list of all inertia-values, e.g.
    inertia_values = list(inertia_dict.values())
    print(inertia_values)

    # reverse the list, IF we want to reduce to the rear wheel (last element in the Torsional Chain)
    ratio_list_rev = ratio_list[::-1]
    for i in range(len(inertia_values)):
        print(math.prod(ratio_list_rev[0:len(ratio_list)-i]))

    # Assuming ratio_list and inertia_values are already defined
    ratio_list_rev = ratio_list[::-1]
    product_list = []  # This will store all the computed products

    for i in range(len(inertia_values)):
        product = math.prod(ratio_list_rev[0:len(ratio_list) - i])
        product_list.append(product)

    # Now product_list contains all the values that were printed
    print(product_list)

    reduced_inertia_list = []

    for i in range(len(product_list)):
        reduced_inertia = inertia_values[i] * product_list[i] ** 2
        reduced_inertia_list.append(reduced_inertia)

    print(reduced_inertia_list)

    # enegry befor is 0.5 * I * omega ** 2 ; using an arbitrary value for omega
    omega = 1000 #rad/s
    E_kin1 = 0.5 * inertia_values[4] * omega ** 2
    print(E_kin1)
    # energy after reduction is 0.5 * I_red * omega_red ** 2 ; using the omega_red = omega * ratio
    omega_red = omega / product_list[4] #rad/s
    E_kin2 = 0.5 * reduced_inertia_list[4] * omega_red ** 2
    print(E_kin2)
    return
