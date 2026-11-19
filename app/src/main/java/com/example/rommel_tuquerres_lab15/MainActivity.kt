package com.example.rommel_tuquerres_lab15

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.OnBackPressedCallback
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.google.mlkit.common.model.LocalModel
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.label.ImageLabel
import com.google.mlkit.vision.label.ImageLabeler
import com.google.mlkit.vision.label.ImageLabeling
import com.google.mlkit.vision.label.custom.CustomImageLabelerOptions
import java.io.IOException
import java.util.Locale

class MainActivity : AppCompatActivity() {

    // One row of assets/test_set/labels.csv
    private data class TestImage(val file: String, val trueLabel: String, val nbPred: String, val nbConf: Float)

    private var testImages: List<TestImage> = emptyList()
    private var current = 0
    private var bitmap: Bitmap? = null
    private var currentItem: TestImage? = null   // null = image picked from the gallery

    private lateinit var labeler: ImageLabeler
    private lateinit var screens: List<ScrollView>
    private lateinit var screenHome: ScrollView
    private lateinit var img: ImageView
    private lateinit var txtFile: TextView
    private lateinit var txtOutput: TextView
    private lateinit var txtSummary: TextView

    // System back button: returns to Home from the other screens
    private val backCallback = object : OnBackPressedCallback(false) {
        override fun handleOnBackPressed() = showScreen(screenHome)
    }

    // Photo picker (no permissions needed)
    private val pickImage = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) {
            bitmap = uriToBitmap(uri)
            bitmap?.let { img.setImageBitmap(it) }
            currentItem = null
            txtFile.text = "Image from gallery (no true label)"
            txtOutput.text = ""
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }
        onBackPressedDispatcher.addCallback(this, backCallback)

        // ---------- Navigation ----------
        screenHome = findViewById(R.id.screenHome)
        val screenModel: ScrollView = findViewById(R.id.screenModel)
        val screenPredict: ScrollView = findViewById(R.id.screenPredict)
        screens = listOf(screenHome, screenModel, screenPredict)

        findViewById<View>(R.id.cardModel).setOnClickListener { showScreen(screenModel) }
        findViewById<View>(R.id.cardPredict).setOnClickListener { showScreen(screenPredict) }
        findViewById<Button>(R.id.btnBackModel).setOnClickListener { showScreen(screenHome) }
        findViewById<Button>(R.id.btnBackPredict).setOnClickListener { showScreen(screenHome) }

        // ---------- Model screen ----------
        assetsToBitmap("training_curves.png")?.let { findViewById<ImageView>(R.id.imgCurves).setImageBitmap(it) }
        assetsToBitmap("confusion_matrix.png")?.let { findViewById<ImageView>(R.id.imgConfusion).setImageBitmap(it) }

        // ---------- Predictions screen ----------
        img = findViewById(R.id.imageToLabel)
        txtFile = findViewById(R.id.txtFile)
        txtOutput = findViewById(R.id.txtOutput)
        txtSummary = findViewById(R.id.txtSummary)

        val localModel = LocalModel.Builder()
            .setAssetFilePath("model.tflite")
            .build()
        val options = CustomImageLabelerOptions.Builder(localModel)
            .setConfidenceThreshold(0.0f)
            .setMaxResultCount(5)
            .build()
        labeler = ImageLabeling.getClient(options)

        testImages = loadTestSet()
        showImage()

        findViewById<Button>(R.id.btnPrev).setOnClickListener {
            if (testImages.isEmpty()) return@setOnClickListener
            current = (current - 1 + testImages.size) % testImages.size
            showImage()
        }

        findViewById<Button>(R.id.btnNext).setOnClickListener {
            if (testImages.isEmpty()) return@setOnClickListener
            current = (current + 1) % testImages.size
            showImage()
        }

        findViewById<Button>(R.id.btnPick).setOnClickListener {
            pickImage.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }

        findViewById<Button>(R.id.btnTest).setOnClickListener {
            val bmp = bitmap ?: return@setOnClickListener
            val t = currentItem
            classify(bmp) { labels ->
                val top = labels.maxBy { it.confidence }
                val sb = StringBuilder()
                sb.append(String.format(Locale.US, "Phone: %s (%.4f)", top.text.trim().uppercase(Locale.ROOT), top.confidence))
                if (t != null) {
                    sb.append("  ").append(mark(top.text.trim() == t.trueLabel))
                    sb.append(String.format(Locale.US, "\nNotebook: %s (%.4f)", t.nbPred, t.nbConf))
                    sb.append("\nTrue label: ").append(t.trueLabel)
                }
                sb.append("\n\nProbabilities:\n")
                for (label in labels) {
                    sb.append(String.format(Locale.US, "%s : %.4f\n", label.text.trim(), label.confidence))
                }
                txtOutput.text = sb.toString()
            }
        }

        findViewById<Button>(R.id.btnAll).setOnClickListener {
            if (testImages.isEmpty()) return@setOnClickListener
            val n = testImages.size
            txtSummary.text = "Classifying $n images..."
            val lines = arrayOfNulls<String>(n)
            var done = 0
            var correct = 0
            var agree = 0
            testImages.forEachIndexed { i, t ->
                val bmp = assetsToBitmap("test_set/${t.file}") ?: return@forEachIndexed
                classify(bmp) { labels ->
                    val top = labels.maxBy { it.confidence }
                    val ok = top.text.trim() == t.trueLabel
                    if (ok) correct++
                    if (top.text.trim() == t.nbPred) agree++
                    lines[i] = String.format(Locale.US, "%s %s · true: %s\n      phone: %s (%.2f) · notebook: %s (%.2f)",
                        mark(ok), t.file.removeSuffix(".jpg"), t.trueLabel, top.text.trim(), top.confidence, t.nbPred, t.nbConf)
                    done++
                    if (done == n) {
                        txtSummary.text = String.format(Locale.US,
                            "Phone accuracy: %d/%d (%.0f %%)\nAgreement with notebook: %d/%d\n\n",
                            correct, n, 100.0 * correct / n, agree, n) + lines.joinToString("\n")
                    }
                }
            }
        }
    }

    private fun showScreen(target: ScrollView) {
        for (s in screens) s.visibility = if (s === target) View.VISIBLE else View.GONE
        target.scrollTo(0, 0)
        backCallback.isEnabled = target !== screenHome
    }

    private fun loadTestSet(): List<TestImage> = try {
        assets.open("test_set/labels.csv").bufferedReader().readLines()
            .drop(1)
            .filter { it.isNotBlank() }
            .map { line ->
                val c = line.trim().split(",")
                TestImage(c[0], c[1], c[2], c[3].toFloat())
            }
    } catch (e: IOException) {
        emptyList()
    }

    private fun showImage() {
        if (testImages.isEmpty()) {
            txtFile.text = "test_set/labels.csv not found in assets"
            return
        }
        val t = testImages[current]
        bitmap = assetsToBitmap("test_set/${t.file}")
        bitmap?.let { img.setImageBitmap(it) }
        currentItem = t
        txtFile.text = "Image ${current + 1}/${testImages.size}: ${t.file} · true label: ${t.trueLabel}"
        txtOutput.text = ""
    }

    private fun classify(bmp: Bitmap, onResult: (List<ImageLabel>) -> Unit) {
        labeler.process(InputImage.fromBitmap(bmp, 0))
            .addOnSuccessListener { labels ->
                if (labels.isEmpty()) txtOutput.text = "No results" else onResult(labels)
            }
            .addOnFailureListener { e -> txtOutput.text = "Error: ${e.message}" }
    }

    // Loads a gallery image, downscaled to about 1024 px to save memory
    private fun uriToBitmap(uri: Uri): Bitmap? = try {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }
        var sample = 1
        while (bounds.outWidth / (sample * 2) >= 1024 && bounds.outHeight / (sample * 2) >= 1024) sample *= 2
        val opts = BitmapFactory.Options().apply { inSampleSize = sample }
        contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, opts) }
    } catch (e: Exception) {
        null
    }

    private fun mark(ok: Boolean) = if (ok) "✅" else "❌"
}

// Load a bitmap from assets
fun Context.assetsToBitmap(fileName: String): Bitmap? {
    return try {
        assets.open(fileName).use { BitmapFactory.decodeStream(it) }
    } catch (e: IOException) {
        null
    }
}