//macro generating undulating filament in 3D
//(in XY plane)
//print("start is here");

//FILAMENT
//length of a segment between nodes, in pixels
nSegmL = 2;
//total number of nodes in the filament
nNodes  = 40;
//number of cilia/filaments
nCilia = 3;

//period of filament undulation (in frames)
nPeriod = 34;
//offset of frequency from the average
offPeriod = newArray(0.0, 10.0, 4.0);
//degree of filament undulation, larger values correspond to more twitching 
flexRig = newArray(0.04,0.02,0.08);
//INTENSITY
//final max brighness of the filament (approximately) 
nSignal = 600;
//constant background
nBG = 100;
//SD of noise
nNoise = 12;
//Starting angle of the filaments in r
startAngle = newArray(0.0, 6.0, 0.2)

//total number of timepoints in the generated data
nTimePoints  = 1024;
//number of z-slices
Zslices = 21;
//width height of the output
outW = 140;
outH = 100;

//Position of the filament's static end
xPos = newArray(20,18,22);
yPos = newArray(50,30,80);
zPos = newArray(Zslices*0.5,Zslices*0.5-2,Zslices*0.5+3);

//Final gaussian blurring SD in each axis (in pixels)
blurSDX = 2.0;
blurSDY = 2.0;
blurSDZ = 2.0;


newImage("HyperStack", "32-bit grayscale-mode", outW, outH, 1, Zslices, nTimePoints);
//newImage("Untitled", "16-bit black", 100, 100, nTimePoints);
setForegroundColor(255, 255, 255);
setBackgroundColor(0, 0, 0);
//run("Fill", "slice");
xpoints = newArray(nNodes);
ypoints = newArray(nNodes);
zpoints = newArray(nNodes);
dAngle = newArray(nCilia);

setBatchMode(false);

for (j=0;j<nCilia;j++)
{
	xpoints[0] = xPos[j];
	ypoints[0] = yPos[j];
	zpoints[0] = zPos[j];
	for(t=1;t<nTimePoints+1;t++)
	{
		showProgress(j*nTimePoints+t, nCilia*nTimePoints);
		Stack.setFrame(t);
		Stack.setSlice(zPos[j]);
		
		currAngle = startAngle[j];
		currAngelo = 0.0;
	
		for (i = 1; i < nNodes; i++)
		{
			dAngle = flexRig[j]*Math.sin(i*PI/(nNodes)+2*PI*t/(nPeriod-offPeriod[j]));	
			dAngelo = flexRig[j]*Math.cos(i*PI/(nNodes)+2*PI*t/(nPeriod-offPeriod[j]));
			currAngle += dAngle;
			currAngelo += dAngelo;
			
			dx = nSegmL*Math.cos(currAngle);
			dy = nSegmL*Math.sin(currAngle);
			dz = 0.1*nSegmL*Math.sin(currAngelo);
			
			xpoints[i] = xpoints[i-1] + dx;
			ypoints[i] = ypoints[i-1] + dy;
			zpoints[i] = zpoints[i-1] + dz;
		}
		//makeSelection("polyline",xpoints, ypoints);
		//run("Fill", "slice");
		// draw filament in z slices
		currentSlice = round(zpoints[0]);
		
		sliceX = newArray(nNodes);
		sliceY = newArray(nNodes);
		
		count = 0;
		
		sliceX[count] = xpoints[0];
		sliceY[count] = ypoints[0];
		count++;
		
		
		for (i=1; i<nNodes; i++)
		{
		    slice = round(zpoints[i]);
		
		    if (slice == currentSlice)
		    {
		        sliceX[count] = xpoints[i];
		        sliceY[count] = ypoints[i];
		        count++;
		    }
		    else
		    {
		        // draw previous section
		        if (count > 1)
		        {
		            Stack.setSlice(currentSlice);
		            sliceX = Array.trim(sliceX, count);
		            sliceY = Array.trim(sliceY, count);
		            		
		            makeSelection("polyline", sliceX, sliceY);
		            run("Draw", "slice");
		        }
		
		        // start new section
		        sliceX = newArray(nNodes);
		        sliceY = newArray(nNodes);
		
		        count = 0;
		        sliceX[count] = xpoints[i];
		        sliceY[count] = ypoints[i];
		        count++;
		
		        currentSlice = slice;
		    }
		}
		
		
		// draw final section
		if (count > 1)
		{
		    Stack.setSlice(currentSlice);
		
		    sliceX = Array.trim(sliceX, count);
		    sliceY = Array.trim(sliceY, count);
		
		    makeSelection("polyline", sliceX, sliceY);
		    run("Draw", "slice");
		    //wait(10);
		}
	}
}
run("Gaussian Blur 3D...", "x="+toString(blurSDX)+" y="+toString(blurSDY)+" z="+toString(blurSDZ));
nSignScale = nSignal/0.04;
run("Multiply...", "value="+toString(nSignScale)+" stack");
setMinAndMax(0, 65500);
run("Add...", "value="+toString(nBG)+" stack");
run("16-bit");
run("Add Specified Noise...", "stack standard="+toString(nNoise));