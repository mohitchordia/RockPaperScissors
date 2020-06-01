package com.occapp.rps;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private static final int PICK_IMAGE = 100 ;
    ImageView GalleryImage;
    Button btnAddImage;

    EditText etResult;
      Uri imageUri;
      Bitmap bitmap;
      Interpreter tflite;
    float[][] vecs = new float[32][16];
    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 128.0f;
    private int channelSize = 3;
    int inputImageWidth = 300;
    int inputImageHeight = 300;
    float[][] tfResult = new float[1][16];
    String[] labels = new String[32];
    String output;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        GalleryImage = findViewById(R.id.gallery_image);
        btnAddImage = findViewById(R.id.add_image_button);

        etResult = findViewById(R.id.result);

        try {
            tflite = new Interpreter(loadModelFile());
        } catch (IOException e) {
            e.printStackTrace();
        }

        btnAddImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
            openGallery();
            }
        });



    }


    private void openGallery() {
        Intent gallery =new  Intent(Intent.ACTION_PICK, MediaStore.Images.Media.INTERNAL_CONTENT_URI);
        startActivityForResult(gallery, PICK_IMAGE);
    }



    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == Activity.RESULT_OK && requestCode == PICK_IMAGE) {
            imageUri = data.getData();
            GalleryImage.setImageURI(imageUri);
            try {
                bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver() , imageUri);
            } catch (IOException e) {
                e.printStackTrace();
            }

            Bitmap resizedImage =Bitmap.createScaledBitmap(bitmap,inputImageWidth,inputImageHeight,true);
            ByteBuffer ModelInput =convertBitmapToByteBuffer(resizedImage);
            tflite.run(ModelInput,tfResult);
            loadVecs();
            loadLabels();
            double[] prob = new double[32];
            for (int i=0; i<32;i++){
                prob[i] = calculateDistance(vecs[i],tfResult[0]);
            }
            int index = calculateIndex(prob);
            if(labels[index].equals("0"))
                output = "Rock";
            else if(labels[index].equals("1"))
                output = "Paper";
            else
                output = "Scissors";
            etResult.setText(output);
        }
    }

    private int calculateIndex(double[] prob) {
        int index=0;
        double min = 100.00;
        for(int i=0;i<prob.length;i++){
            if(prob[i]<min) {
            min = prob[i];
            index = i;
            }
        }

        return index;
    }

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("rock_paper_sci_model.tflite");
        FileInputStream inputStream =  new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declaredLength);
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4  * inputImageWidth * inputImageHeight * channelSize);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[inputImageWidth * inputImageHeight];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < inputImageWidth; ++i) {
            for (int j = 0; j < inputImageHeight; ++j) {
                final int val = intValues[pixel++];
                byteBuffer.putFloat((((val >> 16) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                byteBuffer.putFloat((((val >> 8) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                byteBuffer.putFloat((((val) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
            }
        }
        return byteBuffer;
    }

    private void loadVecs(){
        try {
            BufferedReader TSVFile =
                    new BufferedReader(new InputStreamReader(getAssets().open("vector.txt")));

            String dataRow = TSVFile.readLine();
            int i=0;

            while (dataRow != null){

                String[] dataArray = dataRow.split("\t");
                    int j=0;
                for (String item:dataArray) {
                    vecs[i][j] = Float.valueOf(item.trim());
                    j++;
                }
                i++;
                dataRow = TSVFile.readLine();
            }
            TSVFile.close();
        }
        catch (Exception e){

        }
    }

    private void loadLabels(){

        try {
            BufferedReader TSVFile = new BufferedReader(new InputStreamReader(getAssets().open("rps_labels.tsv")));

            String dataRow = TSVFile.readLine();
            int i=0;
            while (dataRow != null){
                    labels[i++] = dataRow;
                dataRow = TSVFile.readLine();
            }
            TSVFile.close();
        }
        catch (Exception e){

        }
    }
    public static double calculateDistance(float[] array1, float[] array2)
    {
        double Sum = 0.0;
        for(int i=0;i<array1.length;i++) {
            Sum = Sum + Math.pow((array1[i]-array2[i]),2.0);
        }
        return Math.sqrt(Sum);
    }


}