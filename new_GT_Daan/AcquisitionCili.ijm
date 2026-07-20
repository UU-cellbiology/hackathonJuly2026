//assumes that an image generated using 01_flexrod macro is opened,
//simulates individual z-slices over time acquisitions

//maximum possible time shift (in frames), randomly chosen
nMaxShift = 256;
//duration of each z-slice movie (in frames)
nDuration = 256;

//nMaxShift + nDuration should not be larger
//than nTimePoints from the previous macro; typically 1024
saveFolder = getDir("Choose a folder to save the data");
print(saveFolder);

nW = getWidth();
nH = getHeight();
origID = getImageID();

for(x=0;x<10;x++)
//for(x=0;x<3;x++)
{
	selectImage(origID);
	makeLine(x, nH, nW, x);
	run("Reslice [/]...", "output=1.000 slice_count=1 avoid");

	fullresl = getImageID();
	nShift = round(1+random*(nMaxShift-1));
	//print(nShift);
	run("Duplicate...", "duplicate range="+toString(nShift)+"-"+toString(nShift+nDuration-1));
	saveAs("Tiff", saveFolder + "/slice_"+String.pad(x+1, 3) +".tif");
	close();
	selectImage(fullresl);
	close();
}