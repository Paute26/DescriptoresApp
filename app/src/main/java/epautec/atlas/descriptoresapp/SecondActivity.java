package epautec.atlas.descriptoresapp;

import android.annotation.SuppressLint;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.view.View;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import androidx.appcompat.app.AppCompatActivity;

public class SecondActivity extends AppCompatActivity {

    static {
        System.loadLibrary("descriptoresapp");
    }

    private DrawingView drawingView;
    private Button clearButton;
    private Button processButton;
    private ImageView processedImageView;  // Aquí se mostrará el resultado procesado
    private ImageView originalImage;
    public native byte[] processImageNative(byte[] imageData);

    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_second);

        // Inicializar los elementos
        drawingView = findViewById(R.id.drawingView);
        clearButton = findViewById(R.id.clearButton);
        processButton = findViewById(R.id.processButton);
        processedImageView = findViewById(R.id.processedImageView);

        originalImage = findViewById(R.id.originalImageView);
        originalImage.setVisibility(View.GONE);  // Esto la hace invisible pero ocupa el mismo espacio


        // Botón para borrar el dibujo
        clearButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                drawingView.clearCanvas();  // Limpiar el área de dibujo
            }
        });

        // Botón de procesamiento
        processButton.setOnClickListener(v -> {
            // Obtener la imagen como array de bytes
            byte[] imageData = drawingView.getDrawingAsByteArray();

            // Verificar que los datos no sean nulos o vacíos
            if (imageData == null || imageData.length == 0) {
                Log.e("SecondActivity", "Error: imageData es nulo o vacío");
                return;  // Detener el proceso si no hay datos
            }
            Log.d("SecondActivity", "imageData tamaño: " + imageData.length);  // Ver tamaño de imageData

            // Convertir los bytes a Bitmap para mostrar la imagen original
            Bitmap originalBitmap = BitmapFactory.decodeByteArray(imageData, 0, imageData.length);

            // Verificar si la imagen decodificada es nula
            if (originalBitmap == null) {
                Log.e("SecondActivity", "Error: No se pudo decodificar el Bitmap de imageData");
                return;  // Detener el proceso si no se pudo decodificar la imagen
            }

            Log.d("SecondActivity", "Bitmap width: " + originalBitmap.getWidth() + ", height: " + originalBitmap.getHeight());  // Ver tamaño del Bitmap

            //originalImage.setImageBitmap(originalBitmap);

            // Llamar al método nativo para procesar la imagen
            byte[] processedData = processImageNative(imageData);

            // Verificar si la imagen procesada es válida
            if (processedData == null || processedData.length == 0) {
                Log.e("SecondActivity", "Error: processedData es nulo o vacío");
                return;
            }
            Log.d("SecondActivity", "processedData tamaño: " + processedData.length);

            // Convertir los datos procesados a Bitmap y mostrar en ImageView
            Bitmap processedBitmap = BitmapFactory.decodeByteArray(processedData, 0, processedData.length);
            processedImageView.setImageBitmap(processedBitmap);
        });

    }

}