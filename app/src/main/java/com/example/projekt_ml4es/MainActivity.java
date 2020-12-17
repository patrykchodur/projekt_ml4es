package com.example.projekt_ml4es;

import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity {

    ImageView img;
    TextView outputTextView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        img = (ImageView)findViewById(R.id.imageView);
        outputTextView = (TextView)findViewById(R.id.outputTextView);

        try
        {
            // get input stream
            InputStream ims = getAssets().open("test_image.jpg");
            // load image as Drawable
            Drawable d = Drawable.createFromStream(ims, null);
            // set image to ImageView
            img.setImageDrawable(d);
            ims.close();
        }
        catch(IOException ex)
        {
            return;
        }
    }

    public void processClicked(View view) {
        Bitmap bitmap = null;
        try {
            InputStream ims = getAssets().open("test_image.jpg");
            bitmap = BitmapFactory.decodeStream(ims);
        } catch (Exception ex)
        {

        }
        Interpreter interpreter = null;
        try {
            MappedByteBuffer mappedByteBuffer = loadModelFile();
            interpreter = new Interpreter(mappedByteBuffer);
        }
        catch (IOException ex) {
            Log.i("tensor_error", "Could not load model");
            return;
        }
        ByteBuffer input = ByteBuffer.allocateDirect(224 * 224 * 3 * 4).order(ByteOrder.nativeOrder());
        for (int y = 0; y < 224; y++) {
            for (int x = 0; x < 224; x++) {
                int px = bitmap.getPixel(x, y);

                // Get channel values from the pixel value.
                int r = Color.red(px);
                int g = Color.green(px);
                int b = Color.blue(px);

                // Normalize channel values to [-1.0, 1.0]. This requirement depends
                // on the model. For example, some models might require values to be
                // normalized to the range [0.0, 1.0] instead.
                float rf = r / 255.0f;
                float gf = g / 255.0f;
                float bf = b / 255.0f;

                input.putFloat(rf);
                input.putFloat(gf);
                input.putFloat(bf);
            }
        }
        int bufferSize = 4 * java.lang.Float.SIZE / java.lang.Byte.SIZE;
        ByteBuffer modelOutput = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.nativeOrder());
        interpreter.run(input, modelOutput);
        modelOutput.rewind();
        FloatBuffer probabilities = modelOutput.asFloatBuffer();
        String[] labels = {"Fig", "Judastree", "Palm", "Pine"};
        float max_val = 0;
        String max_label = null;
        for (int i = 0; i < probabilities.capacity(); i++) {
                String label = labels[i];
                float probability = probabilities.get(i);
                if (probability > max_val) {
                    max_val = probability;
                    max_label = label;
                }
                Log.i("tensor_output", String.format("%s: %1.4f", label, probability));
        }
        outputTextView.setText(String.format("The photo contains: %s", max_label));

    }

    private MappedByteBuffer loadModelFile() throws IOException{
        AssetFileDescriptor fileDescriptor=this.getAssets().openFd("model_converted.tflite");
        FileInputStream inputStream=new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel=inputStream.getChannel();
        long startOffset=fileDescriptor.getStartOffset();
        long declareLength=fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declareLength);
    }
}