package us.idinfor.opencv;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Vector;

import javax.imageio.ImageIO;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import net.sourceforge.tess4j.TessAPI;
import net.sourceforge.tess4j.Tesseract;
import net.sourceforge.tess4j.TesseractException;
import net.sourceforge.tess4j.util.LoadLibs;

public class Main
{

   public static void main( String[] args ) throws TesseractException, IOException
   {
	   //System.loadLibrary( Core.NATIVE_LIBRARY_NAME );
	   nu.pattern.OpenCV.loadShared();
	   try{
		   new DetectSignalDemo().run();
	   }catch(URISyntaxException e){
		   e.printStackTrace();
	   }
   }
}

//
//Detects signals in an image, draws boxes around them, and writes the results
//to "signalDetection.png".
//
class DetectSignalDemo {

    static int DELAY_CAPTION = 1500;
    static int DELAY_BLUR = 100;
    static int MAX_KERNEL_LENGTH = 31;
    static String windowName = "Filter Demo 1";
	private static Mat roiColor;

	public void run() throws URISyntaxException, TesseractException, IOException {
	 System.out.println("\nRunning DetectSignalDemo");
	
	 // Create a signal detector from the cascade file in the resources
	 // directory.
	final URI xmlUri = getClass().getResource("/cascade.xml").toURI();
    final CascadeClassifier signalDetector =new CascadeClassifier(new File(xmlUri).getAbsolutePath());
    final URI imageUri = getClass().getResource("/81kmh.jpg").toURI();
    final Mat image = Imgcodecs.imread(new File(imageUri).getAbsolutePath(),org.opencv.imgcodecs.Imgcodecs.CV_LOAD_IMAGE_COLOR);
	
	 // Detect signals in the image.
	 MatOfRect signalDetections = new MatOfRect();
	 signalDetector.detectMultiScale(image, signalDetections,1.15,3,0,new Size(20,20),new Size());
	
	 System.out.println(String.format("Detected %s faces", signalDetections.toArray().length));
	
	 // Draw a bounding box around each face.
	 for (Rect rect : signalDetections.toArray()) {		 
		 Imgproc.rectangle(image, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 255, 0));
	 }
	
		System.out.println("Tesseract start ");
		File tess=LoadLibs.extractTessResources("tessdata");
		Tesseract instance = new Tesseract(); //
		instance.setDatapath(tess.getAbsolutePath());
		instance.setLanguage("eng");
		System.setProperty("jna.encoding", "UTF8");
		instance.setOcrEngineMode(TessAPI.TessOcrEngineMode.OEM_DEFAULT);

		String resultPlate = "";
		int i=0;
		for (Rect rect : signalDetections.toArray()) {
			i++;
			BufferedImage letter = matToBufferedImage(image.submat(rect));
			Mat lett = image.submat(rect);
			String result = instance.doOCR(letter);
			System.err.println(" res BufferedImage: "+i+ ": " + result );
			Imgcodecs.imwrite("output/_00XXXZ"+i+".jpg",lett);
			
			 //String filename = "output/_00XXX"+i+".jpg";
			 //System.out.println(String.format("Writing %s", filename));
			 //Imgcodecs.imwrite(filename, image);
			
			
		     //Mat sss = new Mat(image,rect);
		     //Imgcodecs.imwrite("C:\\img\\_00"+i+".jpg", sss  );
			 //filename = "output/_00XXXX"+i+".jpg";
			 //System.out.println(String.format("Writing %s", filename));
			 //Imgcodecs.imwrite(filename, sss);
		    
		    
		    //File input = new File("C:\\img\\_00"+i+".jpg");
		    //result = instance.doOCR (input);
		    //System.err.println(" res: "+i+ ": " + result );
		}		
	 
	 
	 
	 // Save the visualized detection.
	 String filename = "output/signalDetection.png";
	 System.out.println(String.format("Writing %s", filename));
	 Imgcodecs.imwrite(filename, image);
	}




	
/**
 * Converts/writes a Mat into a BufferedImage.
 * 
 * @param matrix Mat of type CV_8UC3 or CV_8UC1
 * @return BufferedImage of type TYPE_3BYTE_BGR or TYPE_BYTE_GRAY
 */
public static BufferedImage matToBufferedImage(Mat matrix) {
    int cols = matrix.cols();
    int rows = matrix.rows();
    int elemSize = (int)matrix.elemSize();
    byte[] data = new byte[cols * rows * elemSize];
    int type;

    matrix.get(0, 0, data);

    switch (matrix.channels()) {
        case 1:
            type = BufferedImage.TYPE_BYTE_GRAY;
            break;

        case 3: 
            type = BufferedImage.TYPE_3BYTE_BGR;

            // bgr to rgb
            byte b;
            for(int i=0; i<data.length; i=i+3) {
                b = data[i];
                data[i] = data[i+2];
                data[i+2] = b;
            }
            break;

        default:
            return null;
    }

    BufferedImage image = new BufferedImage(cols, rows, type);
    image.getRaster().setDataElements(0, 0, cols, rows, data);

    return image;
}

private static Vector<Rect> detectionPlateCharacterContour(Mat roi) {
    Mat contHierarchy = new Mat();
    Mat imageMat = roi.clone();
    Rect rect = null;
    List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
    Imgproc.findContours(imageMat, contours, contHierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
    Vector<Rect> rect_array = new Vector<Rect>();

    for (int i = 0; i < contours.size(); i++) {
        rect = Imgproc.boundingRect(contours.get(i));
        double ratio = 0;

               if(rect.height > rect.width){
            ratio = rect.height/rect.width;

            }else{
                ratio = rect.width/rect.height;

            }
         System.out.println("Ratio of letter: "+ratio);
      double contourarea = Imgproc.contourArea(contours.get(i));
         if 
         (contourarea >= 50 && contourarea <= 1000 && 
        		 ( ratio >= 1 && ratio <= 2)) {
		Imgproc.rectangle(roiColor, rect.br(), rect.tl(), new Scalar(255,0,0));
           rect_array.add(rect);
         }
    }


    contHierarchy.release();
    return rect_array;
}


private static void doTesseractOCR(Vector<Rect> letters, Mat plate){
	System.out.println("Tesseract start ");
Tesseract instance = new Tesseract(); //
instance.setLanguage("ENGLISH");
String resultPlate = "";
	for(int i= 0; i < letters.size(); i++){

		BufferedImage letter = mat2Img(plate.submat(letters.get(i)));
		    try {
			    String result = instance.doOCR(letter);
			    resultPlate += result + " position "+i;
	
		    } catch (TesseractException e) {
		    	System.err.println(e.getMessage());
		    }
	    System.out.println("Tesseract output: "+resultPlate);
	}
}


public static BufferedImage mat2Img(Mat in)
{
    BufferedImage out;
    byte[] data = new byte[320 * 240 * (int)in.elemSize()];
    int type;
    in.get(0, 0, data);

    if(in.channels() == 1)
        type = BufferedImage.TYPE_BYTE_GRAY;
    else
        type = BufferedImage.TYPE_3BYTE_BGR;

    out = new BufferedImage(320, 240, type);

    out.getRaster().setDataElements(0, 0, 320, 240, data);
    return out;
}

public static Mat img2Mat(BufferedImage in)
{
      Mat out;
      byte[] data;
      int r, g, b;

      if(in.getType() == BufferedImage.TYPE_INT_RGB)
      {
          out = new Mat(240, 320, CvType.CV_8UC3);
          data = new byte[320 * 240 * (int)out.elemSize()];
          int[] dataBuff = in.getRGB(0, 0, 320, 240, null, 0, 320);
          for(int i = 0; i < dataBuff.length; i++)
          {
              data[i*3] = (byte) ((dataBuff[i] >> 16) & 0xFF);
              data[i*3 + 1] = (byte) ((dataBuff[i] >> 8) & 0xFF);
              data[i*3 + 2] = (byte) ((dataBuff[i] >> 0) & 0xFF);
          }
      }
      else
      {
          out = new Mat(240, 320, CvType.CV_8UC1);
          data = new byte[320 * 240 * (int)out.elemSize()];
          int[] dataBuff = in.getRGB(0, 0, 320, 240, null, 0, 320);
          for(int i = 0; i < dataBuff.length; i++)
          {
            r = (byte) ((dataBuff[i] >> 16) & 0xFF);
            g = (byte) ((dataBuff[i] >> 8) & 0xFF);
            b = (byte) ((dataBuff[i] >> 0) & 0xFF);
            data[i] = (byte)((0.21 * r) + (0.71 * g) + (0.07 * b)); //luminosity
          }
       }
       out.put(0, 0, data);
       return out;
 } 

/**  
 * Converts/writes a Mat into a BufferedImage.  
 *  
 * @param matrix Mat of type CV_8UC3 or CV_8UC1  
 * @return BufferedImage of type TYPE_3BYTE_BGR or TYPE_BYTE_GRAY  
 */  
public static BufferedImage matToBufferedImage(Mat matrix, BufferedImage bimg)
{
    if ( matrix != null ) { 
        int cols = matrix.cols();  
        int rows = matrix.rows();  
        int elemSize = (int)matrix.elemSize();  
        byte[] data = new byte[cols * rows * elemSize];  
        int type;  
        matrix.get(0, 0, data);  
        switch (matrix.channels()) {  
        case 1:  
            type = BufferedImage.TYPE_BYTE_GRAY;  
            break;  
        case 3:  
            type = BufferedImage.TYPE_3BYTE_BGR;  
            // bgr to rgb  
            byte b;  
            for(int i=0; i<data.length; i=i+3) {  
                b = data[i];  
                data[i] = data[i+2];  
                data[i+2] = b;  
            }  
            break;  
        default:  
            return null;  
        }  

        // Reuse existing BufferedImage if possible
        if (bimg == null || bimg.getWidth() != cols || bimg.getHeight() != rows || bimg.getType() != type) {
            bimg = new BufferedImage(cols, rows, type);
        }        
        bimg.getRaster().setDataElements(0, 0, cols, rows, data);
    } else { // mat was null
        bimg = null;
    }
    return bimg;  
}   

	public static void cannydrawContours(String oriImg, String dstImg, int threshold) {
		
		List<MatOfPoint> contourList = new ArrayList<MatOfPoint>(); 
		
		Mat hierarchy = new Mat();
		final Mat img = Imgcodecs.imread(oriImg);
		Imgproc.cvtColor(img, img, Imgproc.COLOR_BGR2GRAY);
		//
		Imgproc.Canny(img, img, threshold, threshold * 3, 3, true);
	    

		//finding contours
	    Imgproc.findContours(img, contourList, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

	    //Drawing contours on a new image
	    Mat contours = new Mat();
	    contours.create(img.rows(), img.cols(), CvType.CV_8UC3);
	    Random r = new Random();
	    for (int i = 0; i < contourList.size(); i++) {
	        Imgproc.drawContours(contours, contourList, i, new Scalar(r.nextInt(255), r.nextInt(255), r.nextInt(255)), -1);
	    }						
		
		Imgcodecs.imwrite(dstImg, contours);
	}
	
   public static void test(String filename) throws IOException {
       	Mat src = new Mat();
		Mat dst = new Mat();

       src = Imgcodecs.imread(filename, Imgcodecs.IMREAD_COLOR);
       if( src.empty() ) {
           System.out.println("Error opening image");
           System.out.println("Usage: ./Smoothing [image_name -- default ../data/lena.jpg] \n");
           System.exit(-1);
       }
       //if( displayCaption( "Original Image" ) != 0 ) { System.exit(0); }
       dst = src.clone();
       //if( displayDst( DELAY_CAPTION ) != 0 ) { System.exit(0); }
       //if( displayCaption( "Homogeneous Blur" ) != 0 ) { System.exit(0); }
       for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2) {
           Imgproc.blur(src, dst, new Size(i, i), new Point(-1, -1));
           displayDst("C:\\img\\00120blur.jpg",dst);
       }
       //if( displayCaption( "Gaussian Blur" ) != 0 ) { System.exit(0); }
       for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2) {
           Imgproc.GaussianBlur(src, dst, new Size(i, i), 0, 0);
           displayDst("C:\\img\\00120blur2.jpg",dst);
       }
       //if( displayCaption( "Median Blur" ) != 0 ) { System.exit(0); }
       for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2) {
           Imgproc.medianBlur(src, dst, i);
           displayDst("C:\\img\\00120blur3.jpg",dst);
       }
       //if( displayCaption( "Bilateral Blur" ) != 0 ) { System.exit(0); }
       for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2) {
           Imgproc.bilateralFilter(src, dst, i, i * 2, i / 2);
           displayDst("C:\\img\\00120blur4.jpg",dst);
       }       
       System.exit(0);
   }
   /*
   static int displayCaption(String filename, Mat dst) {
	   
       dst = Mat.zeros(src.size(), src.type());
       Imgproc.putText(dst, caption,
               new Point(src.cols() / 4, src.rows() / 2),
               Core.FONT_HERSHEY_COMPLEX, 1, new Scalar(255, 255, 255));
       return displayDst(DELAY_CAPTION);
   }*/

   static int displayDst(String filename, Mat dst) throws IOException {
	   
       byte[] data1 = new byte[dst.rows() * dst.cols() * (int)(dst.elemSize())];
       dst.get(0, 0, data1);
       BufferedImage image1 = new BufferedImage(dst.cols(),dst.rows(), BufferedImage.TYPE_BYTE_GRAY);
       image1.getRaster().setDataElements(0, 0, dst.cols(), dst.rows(), data1);

	   
       File ouptut = new File(filename);
       ImageIO.write(image1, "jpg", ouptut);
       //HighGui.imshow( windowName, dst );
       //int c = HighGui.waitKey( delay );
       //if (c >= 0) { return -1; }
       return 0;
   }

}
