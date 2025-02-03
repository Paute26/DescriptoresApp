package epautec.atlas.descriptoresapp;

import androidx.appcompat.app.AppCompatActivity;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.RadioGroup;
import android.widget.TextView;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

public class MainActivity extends AppCompatActivity {

    // Usar la librería nativa
    static {
        System.loadLibrary("descriptoresapp");
    }

    // Declaración del método nativo
    public native String MomentsHU(byte[] imageData, String csvContent);
    public native String MomentsHUTEST(byte[] imageData, String csvContent);
    public native String MomentsZernike(byte[] imageData, String csvContent);

    private DrawingView drawingView;
    private RadioGroup radioGroup;
    private Button button;
    private TextView resultadoTextView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        radioGroup = findViewById(R.id.radioGroup);
        button = findViewById(R.id.button);
        resultadoTextView = findViewById(R.id.sample_text);
        drawingView = findViewById(R.id.drawingView);

        Button clearButton = findViewById(R.id.clearButton);
        clearButton.setOnClickListener(v -> drawingView.clearCanvas());

        button.setOnClickListener(v -> {
            int selectedId = radioGroup.getCheckedRadioButtonId();

            if (selectedId == R.id.radioButton) {
                clasificarConMomentosHu();
            } else if (selectedId == R.id.radioButton2) {
                clasificarConMomentosZernike();
            } else {
                resultadoTextView.setText("Seleccione un descriptor");
            }
        });
    }

    // Método para leer el archivo CSV desde los assets
    private String loadCsvFromAssets(String path) {
        StringBuilder stringBuilder = new StringBuilder();
        AssetManager assetManager = getAssets();
        try (InputStream inputStream = assetManager.open(path);
             BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream))) {
            String line;
            while ((line = reader.readLine()) != null) {
                stringBuilder.append(line).append("\n");
            }
        } catch (IOException e) {
            Log.e("CSV Load", "Error al leer el archivo CSV", e);
        }
        return stringBuilder.toString();
    }


    private void clasificarConMomentosHu() {
        // Verificar si el usuario ha dibujado algo
        if (drawingView.isEmpty()) {
            resultadoTextView.setText("Por favor, dibuje algo antes de clasificar.");
            return;
        }
        resultadoTextView.setText("Clasificación - Momentos Hu");

        // Obtener la imagen como un array de bytes
        byte[] imageData = drawingView.getDrawingAsByteArray();

        // Cargar el contenido del archivo CSV
        String csvContent = loadCsvFromAssets("moments_of_hu.csv");

        // Llamar al método nativo con la imagen y el contenido del CSV
        String category = MomentsHU(imageData, csvContent);

        // Mostrar el resultado
        resultadoTextView.setText("Categoría: " + category);
    }

    private void clasificarConMomentosZernike() {
        resultadoTextView.setText("Clasificación con Momentos Zernike realizada");
        // Aquí puedes agregar la lógica de clasificación con Momentos Zernike

        // Verificar si el usuario ha dibujado algo
        if (drawingView.isEmpty()) {
            resultadoTextView.setText("Por favor, dibuje algo antes de clasificar.");
            return;
        }
        resultadoTextView.setText("Clasificación - Momentos Hu");

        // Obtener la imagen como un array de bytes
        byte[] imageData = drawingView.getDrawingAsByteArray();

        // Cargar el contenido del archivo CSV
        String csvContent = loadCsvFromAssets("moments_Zernike.csv");

        // Llamar al método nativo con la imagen y el contenido del CSV
        String category = MomentsZernike(imageData, csvContent);

        // Mostrar el resultado
        resultadoTextView.setText("Categoría: " + category);

    }
}
