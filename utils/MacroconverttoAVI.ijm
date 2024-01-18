#@ File (label = "Input directory", style = "directory") input
#@ File (label = "Output directory", style = "directory") output
#@ String (label = "File suffix", value = "E149") mouse
list = getFileList(input);
list = Array.sort(list);
list = Array.deleteIndex(list,0);
newlist = newArray(0);
for (i = 0; i < list.length; i++) {
	if(list[i].contains(mouse))
		newlist = Array.concat(newlist,list[i]);
}
Array.print(newlist)

for (i = 0; i < newlist.length; i++) {
	if(File.isDirectory(input + File.separator + newlist[i])){
		print("Processing: " + newlist[i]);
		currentfiles = getFileList(input + File.separator + newlist[i]);
		currentfiles = Array.sort(currentfiles);
		print(d2s(currentfiles.length,1));
		Array.print(currentfiles);
		timestampfiles = newArray(0);
		for (j = 0; j < currentfiles.length; j++) {
			if(currentfiles[i].contains("time"))
				print(currentfiles[j]);
				timestamppath = input + File.separator + newlist[i] +currentfiles[j];
				timestampfiles = getFileList(input + File.separator + newlist[i] + currentfiles[j]);

		}
		if(timestampfiles.length>0)
			File.copy(timestamppath + timestampfiles[0],output + File.separator + timestampfiles[0]);
		if(timestampfiles.length>0)
			File.copy(timestamppath + timestampfiles[1],output + File.separator + timestampfiles[1]);
		if (currentfiles.length > 1){
			run("Image Sequence...", "open=" + input + File.separator + newlist[i] + File.separator + currentfiles[0] + " sort use");

			print("Saving to: " + output + File.separator + newlist[i]);
			currentsave = newlist[i];
			
			saveAs("AVI", output + File.separator + substring(currentsave,0,currentsave.length-1) +".avi");
		}
	}
}