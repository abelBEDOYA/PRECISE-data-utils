/**
 * QuPath Cropped Exporter (Images + Masks)
 * 
 * Script para exportar imágenes Y máscaras recortadas a la zona con anotaciones.
 * Elimina el espacio vacío innecesario, reduciendo significativamente el tamaño.
 * 
 * Características:
 * - Añade margen configurable
 * - Exporta imagen y máscara con el mismo recorte
 * - Pirámide multinivel coherente
 * - Compresión LZW (lossless)
 * 
 * Requisitos: QuPath 0.6.x o superior
 * 
 * Uso:
 * 1. Abrir proyecto en QuPath
 * 2. Modificar OUTPUT_DIR y parámetros
 * 3. Ejecutar script (Ctrl+R)
 */

import qupath.lib.objects.PathAnnotationObject
import qupath.lib.images.servers.LabeledImageServer
import qupath.lib.images.writers.ome.OMEPyramidWriter
import qupath.lib.common.GeneralTools
import qupath.lib.regions.RegionRequest
import qupath.lib.regions.ImageRegion

// =====================================================
// CONFIGURACIÓN
// =====================================================

// Directorio de salida
def OUTPUT_DIR = "/media/abel/TOSHIBA EXT/export"

// Subdirectorios
def IMAGES_SUBDIR = "images"
def MASKS_SUBDIR = "masks"

// Tamaño de tile para escritura eficiente
def TILE_SIZE = 512

// Niveles de pirámide fijos (potencias de 2)
PYRAMID_DOWNSAMPLES = [1.0d, 2.0d, 4.0d, 8.0d, 16.0d, 32.0d]

// Margen a añadir al bounding box (0.1 = 10%)
def MARGIN_RATIO = 0.1

// Clase a ignorar para calcular el bounding box (Artifact = índice 2 en el array, clase 3)
def IGNORE_CLASS_NAME = "Artifact"

// Crear directorios
def imagesDir = buildFilePath(OUTPUT_DIR, IMAGES_SUBDIR)
def masksDir = buildFilePath(OUTPUT_DIR, MASKS_SUBDIR)
mkdirs(imagesDir)
mkdirs(masksDir)

// =====================================================
// DEFINICIÓN DE CLASES
// =====================================================

def CLASS_NAMES = [
    "Tumor",
    "Bening gland",
    "Artifact",
    "High grade prostatic intraepithelial neoplasia (HGPIN)",
    "Intraductal carcinoma",
    "Atypical intraductal proliferation",
    "Stroma"
    ]

// =====================================================
// FUNCIONES AUXILIARES
// =====================================================

/**
 * Calcula el bounding box combinado de todas las anotaciones,
 * ignorando las de la clase especificada.
 * 
 * @return [x, y, width, height] o null si no hay anotaciones válidas
 */
def calculateAnnotationsBBox(annotations, ignoreClassName) {
    def validAnnotations = annotations.findAll { ann ->
        def pc = ann.getPathClass()
        pc == null || pc.getName() != ignoreClassName
    }
    
    if (validAnnotations.isEmpty()) {
        return null
    }
    
    def minX = Double.MAX_VALUE
    def minY = Double.MAX_VALUE
    def maxX = Double.MIN_VALUE
    def maxY = Double.MIN_VALUE
    
    validAnnotations.each { ann ->
        def roi = ann.getROI()
        def bounds = roi.getBoundsX()  // x
        def boundsY = roi.getBoundsY()  // y
        def boundsW = roi.getBoundsWidth()
        def boundsH = roi.getBoundsHeight()
        
        minX = Math.min(minX, bounds)
        minY = Math.min(minY, boundsY)
        maxX = Math.max(maxX, bounds + boundsW)
        maxY = Math.max(maxY, boundsY + boundsH)
    }
    
    return [
        (int) minX,
        (int) minY,
        (int) (maxX - minX),
        (int) (maxY - minY)
    ]
}

/**
 * Añade margen al bounding box, respetando los límites de la imagen.
 */
def addMargin(bbox, imageWidth, imageHeight, marginRatio) {
    def (x, y, w, h) = bbox
    
    def marginX = (int) (w * marginRatio)
    def marginY = (int) (h * marginRatio)
    
    def newX = Math.max(0, x - marginX)
    def newY = Math.max(0, y - marginY)
    def newMaxX = Math.min(imageWidth, x + w + marginX)
    def newMaxY = Math.min(imageHeight, y + h + marginY)
    
    return [
        newX,
        newY,
        newMaxX - newX,
        newMaxY - newY
    ]
}

/**
 * Devuelve los niveles de downsample fijos para la pirámide.
 * Siempre: 1x, 2x, 4x, 8x, 16x, 32x (potencias de 2)
 */
def getPyramidDownsamples() {
    return PYRAMID_DOWNSAMPLES
}

/**
 * Formatea el tamaño en bytes a una cadena legible
 */
def formatSize(bytes) {
    if (bytes < 1024) return "${bytes} B"
    if (bytes < 1024 * 1024) return String.format("%.2f KB", bytes / 1024.0)
    if (bytes < 1024 * 1024 * 1024) return String.format("%.2f MB", bytes / (1024.0 * 1024.0))
    return String.format("%.2f GB", bytes / (1024.0 * 1024.0 * 1024.0))
}

// =====================================================
// VERIFICAR PROYECTO
// =====================================================

def project = getProject()
if (project == null) {
    print "ERROR: No hay proyecto abierto"
    return
}

print "=========================================="
print "QuPath Cropped Exporter (Images + Masks)"
print "=========================================="
print "Proyecto: ${project.getName()}"
print "Imagenes: ${project.getImageList().size()}"
print "Directorio de salida: ${OUTPUT_DIR}"
print "Margen: ${(MARGIN_RATIO * 100) as int}%"
print "Clase ignorada para bbox: ${IGNORE_CLASS_NAME}"
print "=========================================="
print ""

// =====================================================
// PROCESAR CADA IMAGEN DEL PROYECTO
// =====================================================

def processedCount = 0
def skippedCount = 0
def errorCount = 0
def totalOriginalPixels = 0L
def totalCroppedPixels = 0L

def startTime = System.currentTimeMillis()

project.getImageList().each { entry ->
    
    print "--------------------------------------------"
    print "Procesando: ${entry.getImageName()}"
    
    try {
        def imageData = entry.readImageData()
        def hierarchy = imageData.getHierarchy()
        def server = imageData.getServer()
        def imageName = GeneralTools.stripExtension(entry.getImageName())
        
        def origWidth = server.getWidth()
        def origHeight = server.getHeight()
        def origPixels = (long) origWidth * origHeight
        totalOriginalPixels += origPixels
        
        print "  Original: ${origWidth} x ${origHeight}"
        
        // Obtener anotaciones
        def annotations = hierarchy.getObjects(null, PathAnnotationObject)
        
        if (annotations.isEmpty()) {
            print "  Sin anotaciones - omitiendo"
            skippedCount++
            return
        }
        
        // Calcular bounding box de anotaciones (ignorando Artifact)
        def bbox = calculateAnnotationsBBox(annotations, IGNORE_CLASS_NAME)
        
        if (bbox == null) {
            print "  Sin anotaciones validas (solo Artifact) - omitiendo"
            skippedCount++
            return
        }
        
        print "  Bbox anotaciones: x=${bbox[0]}, y=${bbox[1]}, w=${bbox[2]}, h=${bbox[3]}"
        
        // Añadir margen
        def bboxWithMargin = addMargin(bbox, origWidth, origHeight, MARGIN_RATIO)
        def (cropX, cropY, cropW, cropH) = bboxWithMargin
        
        print "  Bbox con margen: x=${cropX}, y=${cropY}, w=${cropW}, h=${cropH}"
        
        def cropPixels = (long) cropW * cropH
        totalCroppedPixels += cropPixels
        def reduction = ((1.0 - cropPixels / origPixels) * 100) as int
        print "  Reduccion: ${reduction}% (${formatSize(origPixels * 3)} -> ~${formatSize(cropPixels * 3)})"
        
        // Crear región de recorte
        def cropRegion = ImageRegion.createInstance(cropX, cropY, cropW, cropH, 0, 0)
        
        // Niveles de pirámide fijos (1x, 2x, 4x, 8x, 16x, 32x)
        def downsamples = getPyramidDownsamples()
        print "  Niveles de piramide: ${downsamples.size()} (${downsamples.collect { String.format('%.0fx', it) }.join(', ')})"
        
        // =====================================================
        // EXPORTAR IMAGEN RECORTADA
        // =====================================================
        
        def imgOutPath = buildFilePath(imagesDir, "${imageName}.ome.tif")
        
        print "  Exportando imagen..."
        def imgStart = System.currentTimeMillis()
        
        def imgWriter = new OMEPyramidWriter.Builder(server)
            .region(cropRegion)
            .tileSize(TILE_SIZE)
            .downsamples(downsamples as double[])
            .compression(OMEPyramidWriter.CompressionType.LZW)
            .parallelize()
            .build()
        
        imgWriter.writeSeries(imgOutPath)
        
        def imgFile = new File(imgOutPath)
        print String.format("    Imagen: %s (%s) en %.1fs", 
            imgFile.getName(), formatSize(imgFile.length()),
            (System.currentTimeMillis() - imgStart) / 1000.0)
        
        // =====================================================
        // CREAR SERVIDOR DE ETIQUETAS Y EXPORTAR MÁSCARA
        // =====================================================
        
        def labelBuilder = new LabeledImageServer.Builder(imageData)
            .backgroundLabel(0)
            .useAnnotations()
            .multichannelOutput(false)
        
        CLASS_NAMES.eachWithIndex { name, i ->
            labelBuilder.addLabel(name, i + 1)
        }
        
        def labelServer = labelBuilder.build()
        
        def maskOutPath = buildFilePath(masksDir, "${imageName}__mask_multiclass.ome.tif")
        
        print "  Exportando mascara..."
        def maskStart = System.currentTimeMillis()
        
        def maskWriter = new OMEPyramidWriter.Builder(labelServer)
            .region(cropRegion)
            .tileSize(TILE_SIZE)
            .downsamples(downsamples as double[])
            .compression(OMEPyramidWriter.CompressionType.LZW)
            .parallelize()
            .build()
        
        maskWriter.writeSeries(maskOutPath)
        
        def maskFile = new File(maskOutPath)
        print String.format("    Mascara: %s (%s) en %.1fs",
            maskFile.getName(), formatSize(maskFile.length()),
            (System.currentTimeMillis() - maskStart) / 1000.0)
        
        processedCount++
        
        // Liberar memoria
        labelServer.close()
        imageData.getServer().close()
        System.gc()
        
    } catch (Exception e) {
        print "  ERROR: ${e.getMessage()}"
        e.printStackTrace()
        errorCount++
    }
}

// =====================================================
// RESUMEN FINAL
// =====================================================

def totalTime = (System.currentTimeMillis() - startTime) / 1000.0
def totalReduction = totalOriginalPixels > 0 ? 
    ((1.0 - totalCroppedPixels / totalOriginalPixels) * 100) as int : 0

print ""
print "=========================================="
print "PROCESO COMPLETADO"
print "=========================================="
print "Procesadas: ${processedCount}"
print "Omitidas: ${skippedCount}"
print "Errores: ${errorCount}"
print ""
print "Pixeles originales totales: ${String.format('%,d', totalOriginalPixels)}"
print "Pixeles recortados totales: ${String.format('%,d', totalCroppedPixels)}"
print "Reduccion total: ${totalReduction}%"
print ""
print String.format("Tiempo total: %.1f segundos (%.1f minutos)", totalTime, totalTime / 60.0)
print "=========================================="
