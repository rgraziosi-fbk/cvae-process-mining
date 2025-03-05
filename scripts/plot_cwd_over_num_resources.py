import matplotlib.pyplot as plt




# ordine è: sepsis, bpic2012, fines

SEPSIS_KEY = 'Sepsis'
BPIC_KEY = 'Bpic2012'
FINES_KEY = 'Traffic_Fines'

num_res_per_log = {
  SEPSIS_KEY: 25,
  BPIC_KEY: 63,
  FINES_KEY: 148,
}


cwd_roles_per_log = {
  SEPSIS_KEY: 1.5644785628517832,
  BPIC_KEY: (3.2031014537482663 + 3.155424581838153)/2, # (bpic2012_a + bpic2012_b)/2
  FINES_KEY: 11.492038223289885,
}

cwd_resources_per_log = {
  SEPSIS_KEY: 1.5801253350841604,
  BPIC_KEY: (3.437435434998657 + 3.426034616085564)/2,
  FINES_KEY: 11.494356819105917,
}


# plot a line plot
# x axis: number of resources
# y axis: both cwd_roles and cwd_resources (as two lines with two different colors)
plt.figure(figsize=(10, 5))

plt.plot(list(num_res_per_log.values()), list(cwd_roles_per_log.values()), label='CWD_ROLE')
plt.plot(list(num_res_per_log.values()), list(cwd_resources_per_log.values()), label='CWD_RES')

# Define explicit x-ticks and labels
x_ticks = list(num_res_per_log.values())
x_labels = ["25\nSepsis", "63\nBpic2012_a\nBpic2012_b", "112\nTraffic_Fines"]
plt.xticks(x_ticks, x_labels)

plt.xlabel('Number of resources')
plt.ylabel('CWD')
# plt.title('CWD over number of resources')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize='medium')
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()
