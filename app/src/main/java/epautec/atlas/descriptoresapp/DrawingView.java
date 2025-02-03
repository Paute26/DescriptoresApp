package epautec.atlas.descriptoresapp;

        import android.content.Context;
        import android.graphics.Bitmap;
        import android.graphics.Canvas;
        import android.graphics.Color;
        import android.graphics.Paint;
        import android.graphics.Path;
        import android.util.AttributeSet;
        import android.view.MotionEvent;
        import android.view.View;

        import java.io.ByteArrayOutputStream;

public class DrawingView extends View {
    private Paint paint;
    private Path path;

    public DrawingView(Context context, AttributeSet attrs) {
        super(context, attrs);
        paint = new Paint();
        path = new Path();

        // Configuración del pincel
        paint.setColor(Color.BLACK);  // Color del dibujo
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(8f);
        paint.setAntiAlias(true);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        canvas.drawPath(path, paint);  // Dibuja la línea
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        float x = event.getX();
        float y = event.getY();

        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                path.moveTo(x, y);
                return true;
            case MotionEvent.ACTION_MOVE:
                path.lineTo(x, y);
                break;
            case MotionEvent.ACTION_UP:
                break;
        }

        invalidate(); // Redibujar la vista
        return true;
    }
    //Convertir el dibujo en un Bitmap
    public Bitmap getBitmap() {
        Bitmap bitmap = Bitmap.createBitmap(getWidth(), getHeight(), Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(bitmap);
        draw(canvas); // Dibuja el contenido actual en el bitmap
        return bitmap;
    }

    public byte[] getDrawingAsByteArray() {
        Bitmap bitmap = Bitmap.createBitmap(getWidth(), getHeight(), Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(bitmap);
        draw(canvas); // Dibuja el contenido en el bitmap

        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream); // Convertir a PNG en memoria
        return stream.toByteArray();
    }

    public boolean isEmpty() {
        Bitmap bitmap = getBitmap();
        for (int x = 0; x < bitmap.getWidth(); x++) {
            for (int y = 0; y < bitmap.getHeight(); y++) {
                if (bitmap.getPixel(x, y) != Color.WHITE) { // Suponiendo que el fondo es blanco
                    return false; // Hay contenido
                }
            }
        }
        return true; // Está vacío
    }


    // Método para limpiar el dibujo
    public void clearCanvas() {
        path.reset();
        invalidate();
    }
}
