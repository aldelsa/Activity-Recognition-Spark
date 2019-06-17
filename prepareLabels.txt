
# Main Dataset Path 
path_Dataset = "../HAPT Data Set/RawData/"

# File where there are all the labels
file_labels = open(path_Dataset + "labels.txt", "r")

# New file
file_data = open("../tmp/labels_ready.txt", "w+")

# To know when I change of experiment
previousExp = 0

# Index for all lines in labels.txt
index_general = 0

# Index for each experiment in labels.txt
index_local = 0

# Loop each line in labels.txt
for aline in file_labels:

	# Split the columns
	values = aline.split()

	# Check if is a different experiment
	if (int(values[0]) != previousExp):
		index_local = 0

	# If there is not a label to the line I will write a 0 in the file
	if (index_local != int(values[3])):
		for i in range(index_local, int(values[3])):
			file_data.write(values[0] + " " + str(index_general) + " 0\r\n")
			index_local += 1
			index_general += 1

	# If there is a label to the line I will write the correspond label
	for a in range(index_local, int(values[4])):
		file_data.write(values[0] + " " + str(index_general) + " " + values[2] + "\r\n")
		index_local += 1
		index_general += 1

	# I update the previousExp
	previousExp = int(values[0])

# Close the files
file_labels.close()
file_data.close()

