from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from PIL import Image as PILImage


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "pdf"
FIG = ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

PAGE_W, PAGE_H = A4
MARGIN_X = 18 * mm
MARGIN_TOP = 19 * mm
MARGIN_BOTTOM = 17 * mm

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2E6F9E")
TEAL = colors.HexColor("#2A8C82")
LIGHT_BLUE = colors.HexColor("#EAF3F8")
LIGHT_TEAL = colors.HexColor("#EAF6F3")
LIGHT_GREY = colors.HexColor("#F4F6F7")
MID_GREY = colors.HexColor("#66727C")
DARK = colors.HexColor("#1F2933")
ORANGE = colors.HexColor("#E88B3A")


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="DocTitle", fontName="Helvetica-Bold", fontSize=23, leading=27,
    textColor=NAVY, alignment=TA_LEFT, spaceAfter=8,
))
styles.add(ParagraphStyle(
    name="Subtitle", fontName="Helvetica", fontSize=10.5, leading=15,
    textColor=MID_GREY, spaceAfter=10,
))
styles.add(ParagraphStyle(
    name="H1x", fontName="Helvetica-Bold", fontSize=15, leading=19,
    textColor=NAVY, spaceBefore=7, spaceAfter=6,
))
styles.add(ParagraphStyle(
    name="H2x", fontName="Helvetica-Bold", fontSize=11.3, leading=14,
    textColor=BLUE, spaceBefore=5, spaceAfter=4,
))
styles.add(ParagraphStyle(
    name="Bodyx", fontName="Helvetica", fontSize=9.25, leading=13.1,
    textColor=DARK, spaceAfter=5,
))
styles.add(ParagraphStyle(
    name="Smallx", fontName="Helvetica", fontSize=7.8, leading=10.5,
    textColor=MID_GREY, spaceAfter=3,
))
styles.add(ParagraphStyle(
    name="Bulletx", fontName="Helvetica", fontSize=9.1, leading=12.7,
    leftIndent=10, firstLineIndent=-7, bulletIndent=2, textColor=DARK, spaceAfter=3,
))
styles.add(ParagraphStyle(
    name="Calloutx", fontName="Helvetica-Bold", fontSize=10.2, leading=14,
    textColor=NAVY, alignment=TA_CENTER,
))
styles.add(ParagraphStyle(
    name="Captionx", fontName="Helvetica-Oblique", fontSize=7.6, leading=10,
    textColor=MID_GREY, alignment=TA_CENTER, spaceBefore=3, spaceAfter=7,
))
styles.add(ParagraphStyle(
    name="TableHeadx", fontName="Helvetica-Bold", fontSize=8.1, leading=10,
    textColor=colors.white,
))
styles.add(ParagraphStyle(
    name="TableCellx", fontName="Helvetica", fontSize=7.9, leading=10,
    textColor=DARK,
))


def P(text, style="Bodyx"):
    return Paragraph(text, styles[style])


def bullets(items):
    return [P(f"- {item}", "Bulletx") for item in items]


def section(title):
    return P(title, "H1x")


def subsection(title):
    return P(title, "H2x")


def callout(text, color=LIGHT_BLUE):
    t = Table([[P(text, "Calloutx")]], colWidths=[PAGE_W - 2 * MARGIN_X])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), color),
        ("BOX", (0, 0), (-1, -1), 0.7, BLUE),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    return t


def data_table(headers, rows, widths=None):
    cooked = [[P(str(h), "TableHeadx") for h in headers]]
    cooked += [[P(str(v), "TableCellx") for v in row] for row in rows]
    if widths is None:
        widths = [(PAGE_W - 2 * MARGIN_X) / len(headers)] * len(headers)
    t = Table(cooked, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C8D1D8")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def figure(filename, caption, max_w=170 * mm, max_h=95 * mm):
    path = FIG / filename
    if not path.exists():
        return [P(f"Figura no encontrada: {filename}", "Smallx")]
    with PILImage.open(path) as im:
        w, h = im.size
    scale = min(max_w / w, max_h / h)
    img = Image(str(path), width=w * scale, height=h * scale)
    img.hAlign = "CENTER"
    return [img, P(caption, "Captionx")]


def flow_box(items):
    cells = []
    for i, item in enumerate(items):
        cells.append(P(f"<b>{item}</b>", "TableCellx"))
        if i < len(items) - 1:
            cells.append(P("->", "TableCellx"))
    widths = []
    usable = PAGE_W - 2 * MARGIN_X
    arrows = len(items) - 1
    item_w = (usable - arrows * 9 * mm) / len(items)
    for i in range(len(cells)):
        widths.append(item_w if i % 2 == 0 else 9 * mm)
    t = Table([cells], colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_TEAL),
        ("BOX", (0, 0), (-1, -1), 0.6, TEAL),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def page_decor(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D5DEE5"))
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN_X, 13 * mm, PAGE_W - MARGIN_X, 13 * mm)
    canvas.setFont("Helvetica", 7.4)
    canvas.setFillColor(MID_GREY)
    canvas.drawString(MARGIN_X, 8.5 * mm, "Taller B5-T1 - Generacion de datos financieros sinteticos")
    canvas.drawRightString(PAGE_W - MARGIN_X, 8.5 * mm, f"Pagina {doc.page}")
    canvas.restoreState()


class GuideDoc(BaseDocTemplate):
    def __init__(self, filename, **kw):
        super().__init__(filename, pagesize=A4, leftMargin=MARGIN_X, rightMargin=MARGIN_X,
                         topMargin=MARGIN_TOP, bottomMargin=MARGIN_BOTTOM, **kw)
        frame = Frame(MARGIN_X, MARGIN_BOTTOM, PAGE_W - 2 * MARGIN_X,
                      PAGE_H - MARGIN_TOP - MARGIN_BOTTOM, id="normal")
        self.addPageTemplates(PageTemplate(id="guide", frames=[frame], onPage=page_decor))


def title_block(number, title, purpose, notebook_name):
    return [
        P(f"Notebook {number}", "Subtitle"),
        P(title, "DocTitle"),
        P(purpose, "Subtitle"),
        callout(f"Idea central: {purpose}", LIGHT_BLUE),
        Spacer(1, 7),
        P(f"Fuente analizada: notebooks/{notebook_name}", "Smallx"),
    ]


def build_00():
    story = title_block(
        "00", "Descarga y consolidacion de datos",
        "Crear los datasets reales de los que depende todo el pipeline",
        "00_descarga_datos.ipynb",
    )
    story += [section("Que problema resuelve"), P(
        "El proyecto necesita dos escalas temporales incompatibles: retornos diarios con decadas de historia y "
        "microestructura intradia disponible solo en años recientes. Este notebook integra ambas fuentes, define "
        "dos universos de bancos y prepara un pool conjunto limpio para los modelos generativos."
    )]
    story += [subsection("Entradas")]
    story += bullets([
        "Norgate DuckDB: precios diarios ajustados a retorno total entre 1990-01-02 y 2026-05-29.",
        "EODHD: barras OHLCV de 5 minutos en UTC, cacheadas por ticker.",
        "Configuracion: 25 bancos para el predictor y 150 para entrenar los generadores.",
        "Credencial EODHD leida desde datos/APkey, sin imprimirla.",
    ])
    story += [Spacer(1, 5), flow_box(["Precios diarios", "Retornos log", "Features 5m", "Pool conjunto"])]
    story += [section("Por que hay dos universos")]
    story += [data_table(
        ["Universo", "Tamaño", "Requisito", "Uso"],
        [
            ["Predictor", "25 bancos", "Historia casi completa desde 1990", "Target y ventanas supervisadas de 30 años"],
            ["Generadores", "150 bancos", "Cobertura reciente suficiente", "Aprender una distribucion conjunta mas rica"],
        ],
        [32 * mm, 23 * mm, 58 * mm, 60 * mm],
    )]
    story += [PageBreak(), section("Transformaciones principales")]
    story += [subsection("1. Retornos diarios")]
    story += [P(
        "Se calcula r(t) = log(P(t)) - log(P(t-1)) usando el cierre ajustado a retorno total. Para los 25 bancos "
        "se conserva una matriz densa y sincronizada; para los 150 se permiten NaN por altas, bajas o coberturas parciales."
    )]
    story += [subsection("2. Features intradia por sesion")]
    story += [data_table(
        ["Variable", "Definicion", "Interpretacion"],
        [
            ["realized_vol", "Raiz de la suma de retornos 5m al cuadrado", "Volatilidad realizada del dia"],
            ["open_30m_ret", "Retorno de las primeras 6 barras", "Movimiento de apertura"],
            ["close_30m_ret", "Retorno de las ultimas 6 barras", "Movimiento de cierre"],
            ["hl_range", "(max(high)-min(low))/open", "Amplitud intradia"],
            ["n_bars", "Numero de timestamps por sesion", "Control de cobertura"],
        ],
        [35 * mm, 72 * mm, 66 * mm],
    )]
    story += [subsection("3. Controles de calidad")]
    story += bullets([
        "Una sesion necesita al menos 40 barras; un ticker, al menos 60 sesiones validas.",
        "Se eliminan filas con NaN al cruzar retorno y features.",
        "Se aplican limites de cordura a retornos, volatilidad y rango intradia.",
        "El pool final no contiene NaN ni infinitos.",
    ])
    story += [section("Dimensiones observadas")]
    story += [data_table(
        ["Objeto", "Dimension / volumen", "Periodo"],
        [
            ["Retornos predictor", "7.865 x 25", "1990-01-03 a 2026-05-29"],
            ["Retornos generador", "502 x 150; 2,1% NaN", "2024-05-29 a 2026-05-29"],
            ["Barras EODHD", "16.369.276 filas", "2020-11-02 a 2026-08-28"],
            ["Features intradia", "208.386 filas; 149 tickers", "2020-11-02 a 2026-08-28"],
            ["Pool condicional", "53.282 x 5; 148 tickers", "2024-05-30 a 2026-05-29"],
        ],
        [51 * mm, 60 * mm, 62 * mm],
    )]
    story += [PageBreak(), section("Salidas y lectura critica")]
    story += [subsection("Archivos que deja preparados")]
    story += bullets([
        "returns_predictor.parquet y returns_generator.parquet.",
        "intraday_features_real.parquet.",
        "conditional_pool.npy y conditional_pool_meta.parquet.",
        "Tablas de cobertura diaria e intradia en reports/tables.",
    ])
    story += [subsection("Hallazgos de calidad")]
    story += bullets([
        "Los 150 tickers tienen algun dato intradia, pero la cobertura es desigual.",
        "El 81,1% de las filas crudas tiene OHLC completo; el filtrado posterior protege el pool final.",
        "Solo 5 de 53.287 candidatos se descartan por valores implausibles (0,01%).",
        "El pool final queda completo y trazable por ticker y fecha.",
    ])
    story += [callout(
        "Resultado: una base real, limpia y alineada para comparar cuatro generadores bajo el mismo problema.",
        LIGHT_TEAL,
    ), Spacer(1, 8)]
    story += [subsection("Precaucion para reproducir")]
    story += [P(
        "En el equipo actual, config.py espera datos dentro del repositorio, pero la carpeta real datos esta un nivel "
        "por encima. Las salidas visibles del notebook proceden de una ejecucion anterior en Windows. Antes de "
        "relanzarlo hay que alinear esa ruta, sin publicar APkey."
    )]
    story += [subsection("Conexion con el notebook 01"), P(
        "El notebook 01 toma las barras cacheadas y demuestra que la estructura intradia contiene señal que no esta "
        "completamente representada por el retorno diario."
    )]
    return story


def build_01():
    story = title_block(
        "01", "EDA de la distribucion intradia",
        "Demostrar que la volatilidad y el volumen tienen estructura temporal y aportan informacion adicional",
        "01_eda_intradia.ipynb",
    )
    story += [section("Pregunta de investigacion"), P(
        "Si la volatilidad intradia fuera una transformacion determinista del retorno diario, reconstruirla con datos "
        "sinteticos no aportaria valor. El notebook contrasta esa hipotesis usando JPM, banco grande y liquido, y "
        "GBCI, banco regional de menor tamaño."
    )]
    story += [subsection("Muestra")]
    story += [data_table(
        ["Ticker", "Barras 5m", "Periodo", "Rol comparativo"],
        [
            ["JPM", "114.928", "2020-11-02 a 2026-08-28", "Banco money-center"],
            ["GBCI", "114.924", "2020-11-02 a 2026-08-28", "Banco regional"],
        ],
        [25 * mm, 31 * mm, 63 * mm, 54 * mm],
    )]
    story += [section("Analisis realizado")]
    story += bullets([
        "Agrupa retornos de 5 minutos por hora local de Nueva York.",
        "Calcula retorno absoluto medio, desviacion y volumen medio por franja.",
        "Resume la distribucion diaria de realized_vol.",
        "Cruza |retorno close-to-close| con volatilidad realizada en fechas comunes.",
    ])
    story += [PageBreak(), section("Perfil a lo largo del dia")]
    story += figure("01_perfil_intradia_volatilidad.png",
                    "Volatilidad media por franja: maxima en apertura y cierre, minima a mediodia.", max_h=82 * mm)
    story += figure("01_perfil_intradia_volumen.png",
                    "El volumen reproduce la misma forma de U, con un repunte claro en la subasta de cierre.", max_h=82 * mm)
    story += [P(
        "La forma de U aparece en ambos bancos, lo que sugiere un patron microestructural estable. La diferencia esta "
        "en la escala: el banco regional presenta mayor variabilidad relativa."
    )]
    story += [PageBreak(), section("Volatilidad diaria y valor informativo")]
    story += figure("01_distribucion_realized_vol.png",
                    "Distribucion de realized_vol: GBCI esta desplazado hacia niveles mas altos que JPM.", max_h=76 * mm)
    story += [data_table(
        ["Ticker", "Media RV", "Mediana RV", "Maximo"],
        [["JPM", "0,0113", "0,0103", "0,0810"], ["GBCI", "0,0176", "0,0164", "0,0846"]],
        [43 * mm, 43 * mm, 43 * mm, 44 * mm],
    ), Spacer(1, 7)]
    story += figure("01_relacion_retorno_vs_realized_vol.png",
                    "Relacion moderada entre |retorno diario| y volatilidad realizada.", max_h=75 * mm)
    story += [callout(
        "Conclusion: retorno y volatilidad estan relacionados, pero no son redundantes. La trayectoria intradia añade informacion.",
        LIGHT_TEAL,
    )]
    story += [subsection("Limitaciones")]
    story += bullets([
        "La demostracion visual usa solo dos bancos; no prueba universalidad por si sola.",
        "La correlacion lineal no captura todas las dependencias ni implica causalidad.",
        "El EDA usa toda la cache 2020-2026, mientras el entrenamiento final respeta cortes temporales mas estrictos.",
    ])
    story += [subsection("Conexion con el notebook 02"), P(
        "Justificada la utilidad potencial de las features intradia, el siguiente notebook aprende su distribucion "
        "conjunta con el retorno mediante cuatro modelos generativos."
    )]
    return story


def build_02():
    story = title_block(
        "02", "Entrenamiento de cuatro modelos generativos",
        "Generar muestras sinteticas comparables de retorno y microestructura sin contaminar validacion ni test",
        "02_modelos_generativos.ipynb",
    )
    story += [section("Datos y separacion")]
    story += [data_table(
        ["Paso", "Muestras", "Uso"],
        [
            ["Pool completo", "53.282", "Datos reales alineados del notebook 00"],
            ["Tras excluir val/test", "25.183", "Solo fechas anteriores a 2025-06-01"],
            ["Train generativo", "22.665", "Ajuste de los modelos"],
            ["Referencia real", "2.518", "Comparacion distribucional"],
        ],
        [49 * mm, 35 * mm, 89 * mm],
    )]
    story += [P(
        "Cada fila tiene cinco variables: log_return, realized_vol, open_30m_ret, close_30m_ret y hl_range. "
        "Los modelos son incondicionales: aprenden la distribucion conjunta y generan pares nuevos. El condicionamiento "
        "por retorno historico se reserva para el notebook 03."
    )]
    story += [section("Los cuatro generadores")]
    story += [data_table(
        ["Modelo", "Mecanismo", "Configuracion clave", "Salida"],
        [
            ["Ruido", "Remuestreo real + ruido gaussiano", "sigma 0,15 relativo", "50.000 x 5"],
            ["Gaussiano", "Normal multivariante", "Covarianza Ledoit-Wolf; shrinkage 0,007", "50.000 x 5"],
            ["RBIG", "Gaussianizacion marginal + rotaciones", "20 iteraciones; grid 400", "50.000 x 5"],
            ["GAN", "Generador y discriminador densos", "1.000 epochs; latent 32; BCE", "50.000 x 5"],
        ],
        [29 * mm, 49 * mm, 64 * mm, 31 * mm],
    )]
    story += [PageBreak(), section("Convergencia y diagnostico")]
    story += figure("02_rbig_convergencia.png",
                    "RBIG converge cuando disminuye el exceso de curtosis respecto a una Normal conjunta.", max_h=67 * mm)
    story += figure("02_gan_convergencia.png",
                    "Evolucion de las perdidas adversariales del generador y discriminador.", max_h=67 * mm)
    story += [P(
        "Ruido y Gaussiano no tienen una loss iterativa: el primero no optimiza y el segundo se ajusta en forma cerrada. "
        "Su evaluacion se basa en fidelidad marginal y estructura de dependencia."
    )]
    story += [PageBreak(), section("Calidad de las muestras sinteticas")]
    story += figure("02_real_vs_sintetico_por_generador.png",
                    "Comparacion de histogramas reales y sinteticos para las cinco variables.", max_h=105 * mm)
    story += [data_table(
        ["Generador", "Distancia Frobenius de correlacion", "Lectura"],
        [
            ["Ruido", "0,386", "Mejor preservacion de correlaciones"],
            ["Gaussiano", "0,404", "Muy proximo al modelo de ruido"],
            ["RBIG", "0,480", "Algo peor en dependencia lineal"],
            ["GAN", "1,820", "Peor resultado en esta configuracion"],
        ],
        [41 * mm, 58 * mm, 74 * mm],
    ), Spacer(1, 7)]
    story += [callout(
        "La metrica distribucional no decide todavia el mejor modelo: la prueba decisiva es su utilidad para el predictor.",
        LIGHT_TEAL,
    )]
    story += [subsection("Controles y salidas")]
    story += bullets([
        "realized_vol y hl_range se recortan a cero para respetar su soporte fisico.",
        "Se guardan synthetic_pool_noise, gaussian, rbig y gan en formato NPY.",
        "El mismo mecanismo posterior de condicionamiento se aplicara a los cuatro.",
    ])
    story += [subsection("Conexion con el notebook 03"), P(
        "Los cuatro pools se convierten en historias intradia plausibles para cada retorno diario real de 1996-2024."
    )]
    return story


def build_03():
    story = title_block(
        "03", "Backfill condicional y dataset supervisado",
        "Reconstruir 28 años de volatilidad intradia plausible sin inventar el retorno diario",
        "03_backfill_condicional.ipynb",
    )
    story += [section("Idea del backfill")]
    story += [P(
        "El retorno diario historico de cada banco ya es real y conocido gracias a Norgate. Lo ausente antes de mayo "
        "de 2024 son las features intradia. Para cada retorno historico, el notebook selecciona una muestra sintetica "
        "cuyo retorno generado sea cercano y usa sus features como reconstruccion plausible."
    ), flow_box(["Retorno real del dia", "80 vecinos sinteticos", "Kernel gaussiano", "Features elegidas"])]
    story += [subsection("Por que no hay look-ahead")]
    story += bullets([
        "La consulta usa el retorno del mismo dia historico, no el retorno del dia siguiente.",
        "El target del predictor nunca participa en el emparejamiento.",
        "Los pools generativos fueron entrenados sin validacion ni test.",
        "No es pandas bfill: no propaga valores futuros hacia el pasado.",
    ])
    story += [section("Entradas")]
    story += [data_table(
        ["Entrada", "Dimension", "Contenido"],
        [
            ["returns_predictor", "7.387 x 25 desde 1996", "Retornos diarios reales"],
            ["Features reales", "25/25 bancos", "Microestructura desde 2024"],
            ["4 pools sinteticos", "50.000 x 5 cada uno", "Muestras de cada generador"],
        ],
        [45 * mm, 46 * mm, 82 * mm],
    )]
    story += [PageBreak(), section("Construccion de la historia completa")]
    story += figure("03_backfill_serie_temporal_JPM.png",
                    "JPM: tramo sintetico en gris y tramo real en color para tres generadores.", max_h=120 * mm)
    story += [P(
        "Cada generador produce un panel de 7.387 fechas por 25 bancos. Por ticker, 6.885 dias son sinteticos y el "
        "tramo reciente utiliza mediciones reales. La visualizacion no pretende acertar la volatilidad historica dia "
        "a dia, sino conservar nivel, variabilidad y respuesta a retornos extremos."
    )]
    story += [subsection("Continuidad en mayo de 2024")]
    story += [data_table(
        ["Generador", "Ratio sintetico / real", "Desv. entre bancos"],
        [
            ["Ruido", "1,242", "0,114"],
            ["RBIG", "1,244", "0,121"],
            ["Gaussiano", "1,279", "0,160"],
            ["GAN", "1,379", "0,151"],
        ],
        [49 * mm, 61 * mm, 63 * mm],
    )]
    story += [PageBreak(), section("Ventanas para el predictor")]
    story += [data_table(
        ["Objeto", "Forma", "Significado"],
        [
            ["X", "7.326 x 60 x 50", "60 dias; retorno + volatilidad para 25 bancos"],
            ["Y", "7.326 x 25", "Retorno del dia siguiente para cada banco"],
            ["idx", "7.326 fechas", "Ultimo dia observado por cada ventana"],
            ["is_synthetic", "7.326 booleanos", "Indica uso de historia sintetica"],
        ],
        [40 * mm, 44 * mm, 89 * mm],
    )]
    story += [subsection("Lectura de los canales")]
    story += [P(
        "Los primeros 25 canales de X son retornos diarios reales. Los 25 restantes son volatilidades realizadas: "
        "sinteticas antes del corte y reales despues. Y contiene solo retornos, nunca volatilidad."
    )]
    story += [subsection("Salidas")]
    story += bullets([
        "dataset_noise.npz, dataset_gaussian.npz, dataset_rbig.npz y dataset_gan.npz.",
        "Tabla 03_continuidad_empalme.csv.",
        "Figura temporal de JPM para validacion visual.",
    ])
    story += [callout(
        "Resultado: cuatro problemas supervisados identicos salvo por el generador que reconstruyo la volatilidad historica.",
        LIGHT_TEAL,
    )]
    story += [subsection("Limitacion metodologica")]
    story += [P(
        "La feature sintetica se deriva parcialmente del retorno del mismo dia. Es plausible y causal, pero contiene "
        "menos informacion independiente que una medicion intradia real."
    )]
    story += [subsection("Conexion con el notebook 04"), P(
        "El siguiente notebook mantiene fijo el test real y compara arquitectura, profundidad historica y porcentaje sintetico."
    )]
    return story


def build_04():
    story = title_block(
        "04", "Entrenamiento del predictor del dia siguiente",
        "Elegir una arquitectura sin mirar el test y medir el efecto de añadir datos sinteticos",
        "04_entrenamiento_predictor.ipynb",
    )
    story += [section("Diseño experimental")]
    story += [data_table(
        ["Particion", "Fechas", "Muestras", "Uso"],
        [
            ["Train real", "2024-05-29 a 2025-05-31", "252 iniciales", "Seleccion de arquitectura"],
            ["Validacion", "2025-06-01 a 2025-11-30", "126", "Seleccion y early stopping"],
            ["Test", "Desde 2025-12-01", "122", "Evaluacion final comun"],
        ],
        [35 * mm, 52 * mm, 31 * mm, 55 * mm],
    )]
    story += [P(
        "Todas las entradas tienen forma (60, 50): 60 dias de historia y dos canales para cada uno de 25 bancos. "
        "La salida tiene 25 retornos del dia siguiente."
    )]
    story += [subsection("Entrenamiento")]
    story += bullets([
        "Loss: MAE, robusta frente a las colas pesadas de los retornos.",
        "Batch size: 64.",
        "Techo: 300 epochs para arquitectura y 500 para las rejillas.",
        "EarlyStopping con paciencia 100 y restauracion de mejores pesos.",
    ])
    story += [section("Seleccion de arquitectura")]
    story += [data_table(
        ["Familia", "Variantes"],
        [
            ["Referencias", "Baseline y regresion lineal"],
            ["Densa", "Capas 128 y 64"],
            ["CNN", "1 bloque o 3 bloques convolucionales"],
            ["RNN", "LSTM de 1 capa o 2 capas"],
        ],
        [47 * mm, 126 * mm],
    )]
    story += [PageBreak(), section("Resultado de la seleccion")]
    story += figure("04_loss_curvas_arquitecturas.png",
                    "Curvas de entrenamiento y validacion de las arquitecturas candidatas.", max_h=105 * mm)
    story += [data_table(
        ["Modelo", "MAE validacion", "MAE test", "Parametros"],
        [
            ["RNN 2 capas", "0,011485", "0,011956", "143.681"],
            ["RNN 1 capa", "0,011518", "0,011867", "38.465"],
            ["CNN 3 bloques", "0,011563", "0,011811", "150.273"],
            ["CNN 1 bloque", "0,011594", "0,011995", "197.889"],
            ["Densa", "0,011813", "0,012074", "394.009"],
        ],
        [46 * mm, 44 * mm, 41 * mm, 42 * mm],
    ), Spacer(1, 6)]
    story += [P(
        "Aunque la RNN de dos capas obtiene el minimo, cinco redes estan dentro de un error estandar. Se aplica la "
        "regla de una desviacion estandar y se elige la opcion mas simple: RNN de una capa. El test no se usa para elegir."
    )]
    story += [PageBreak(), section("Dos rejillas complementarias")]
    story += [subsection("Años de historia sintetica")]
    story += [P(
        "Se comparan 0, 7, 14, 21 y 28 años de backfill para cada generador. Esta rejilla responde a la pregunta "
        "financiera: cuanto contexto historico recuperamos."
    )]
    story += figure("04_mae_vs_profundidad.png",
                    "MAE de test frente a profundidad historica sintetica.", max_h=68 * mm)
    story += [subsection("Porcentaje sintetico")]
    story += [P(
        "Se comparan 0%, 25%, 50%, 75%, 90% y 100%. Mantiene todas las ventanas reales y añade ventanas sinteticas "
        "hasta alcanzar la proporcion. El 100% elimina el ancla real."
    )]
    story += figure("04_precision_direccional_vs_porcentaje.png",
                    "Acierto direccional para cada generador y porcentaje sintetico.", max_h=68 * mm)
    story += [PageBreak(), section("Metricas, controles y salidas")]
    story += [data_table(
        ["Metrica", "Que mide", "Lectura"],
        [
            ["MAE", "Error absoluto medio", "Magnitud del error en retornos"],
            ["MSE", "Error cuadratico medio", "Penaliza especialmente extremos"],
            ["Directional accuracy", "Proporcion de signos correctos", "50% equivale aproximadamente al azar"],
        ],
        [45 * mm, 62 * mm, 66 * mm],
    )]
    story += [subsection("Evitar comparaciones sesgadas")]
    story += bullets([
        "Mismo test real para todos los modelos.",
        "Misma arquitectura ganadora y reinicializacion de pesos.",
        "Validacion y test nunca se usan para entrenar generadores.",
        "Desglose MAE por banco para que los mas volatiles no dominen la media.",
        "Curvas de loss guardadas para todos los entrenamientos iterativos.",
    ])
    story += [subsection("Salidas")]
    story += bullets([
        "04_comparacion_arquitecturas.csv.",
        "04_resultados_rejilla_profundidad.csv y 04_resultados_rejilla_porcentaje.csv.",
        "04_mae_por_banco.csv y pickles consolidados.",
        "Graficas de MAE, MSE, precision direccional y convergencia.",
    ])
    story += [callout(
        "Este notebook produce la evidencia experimental; el 05 la resume y evita sobreinterpretar rankings inestables.",
        LIGHT_TEAL,
    )]
    return story


def build_05():
    story = title_block(
        "05", "Analisis e interpretacion de resultados",
        "Convertir las rejillas del notebook 04 en conclusiones financieras y metodologicas defendibles",
        "05_analisis_resultados.ipynb",
    )
    story += [section("Que hace")]
    story += [P(
        "No entrena modelos. Lee tablas de calidad generativa, continuidad del backfill y rendimiento predictivo; "
        "construye las figuras finales y contrasta tres preguntas: si los sinteticos ayudan, cuanto añadir y si existe "
        "un generador consistentemente mejor."
    )]
    story += [subsection("Referencia")]
    story += [callout(
        "Solo datos reales: MAE 0,01191 y precision direccional cercana al 50%.",
        LIGHT_BLUE,
    ), Spacer(1, 7)]
    story += [section("Resultados por profundidad")]
    story += [data_table(
        ["Años sinteticos", "Ruido", "Gaussiano", "RBIG", "GAN"],
        [
            ["0", "0,01191", "0,01191", "0,01191", "0,01191"],
            ["7", "0,01255", "0,01199", "0,01242", "0,01186"],
            ["14", "0,01229", "0,01179", "0,01222", "0,01181"],
            ["21", "0,01231", "0,01178", "0,01182", "0,01180"],
            ["28", "0,01186", "0,01187", "0,01174", "0,01184"],
        ],
        [37 * mm, 34 * mm, 34 * mm, 34 * mm, 34 * mm],
    )]
    story += [P(
        "A maxima profundidad, RBIG presenta el menor MAE en esta ejecucion: 0,01174, una mejora aproximada del "
        "1,36% frente a la referencia. La diferencia entre generadores es pequeña."
    )]
    story += [PageBreak(), section("Efecto de la cantidad sintetica")]
    story += figure("05_resultado_final_mae.png",
                    "MAE final al añadir años de historia sintetica.", max_h=78 * mm)
    story += figure("05_acierto_vs_porcentaje.png",
                    "Acierto direccional medio de los cuatro generadores por porcentaje sintetico.", max_h=78 * mm)
    story += [data_table(
        ["Porcentaje sintetico", "Acierto medio", "Mejora vs. base"],
        [
            ["25%", "51,92%", "+1,82 puntos"],
            ["50%", "51,72%", "+1,62 puntos"],
            ["75%", "51,41%", "+1,31 puntos"],
            ["90%", "53,54%", "+3,44 puntos"],
            ["100%", "52,27%", "+2,17 puntos"],
        ],
        [58 * mm, 52 * mm, 63 * mm],
    )]
    story += [P(
        "La curva sugiere que añadir sinteticos ayuda y que conservar algo de dato real como ancla es conveniente: "
        "el pico aparece en 90% y cae al 100%. No es monotona y no constituye una prueba estadistica definitiva."
    )]
    story += [PageBreak(), section("Calidad generativa frente a utilidad")]
    story += figure("05_calidad_generador_vs_mae.png",
                    "La fidelidad de correlacion no ordena de forma clara el rendimiento predictivo.", max_h=82 * mm)
    story += [data_table(
        ["Generador", "Dist. correlacion", "MAE con 28 años", "Ranking"],
        [
            ["Ruido", "0,386", "0,011856", "3"],
            ["Gaussiano", "0,404", "0,011867", "4"],
            ["RBIG", "0,480", "0,011745", "1"],
            ["GAN", "1,820", "0,011842", "2"],
        ],
        [39 * mm, 45 * mm, 52 * mm, 37 * mm],
    )]
    story += [P(
        "El modelo que mejor reproduce la correlacion real no es el que obtiene el menor MAE. La fidelidad "
        "distribucional medida en el notebook 02 no predice por si sola la utilidad aguas abajo."
    )]
    story += [PageBreak(), section("La conclusion honesta")]
    story += [subsection("Que parece sostenerse")]
    story += bullets([
        "La referencia puramente real tiene capacidad direccional cercana a una moneda al aire.",
        "Añadir historia sintetica puede mejorar modestamente MAE y acierto direccional.",
        "La media de los generadores muestra un pico alrededor del 90% sintetico.",
        "No conviene eliminar por completo los datos reales del entrenamiento.",
    ])
    story += [subsection("Que no puede afirmarse")]
    story += bullets([
        "No hay evidencia estable de que RBIG, GAN, Gaussiano o Ruido sea universalmente superior.",
        "Una repeticion de la rejilla movio configuraciones hasta 6,36 puntos y produjo correlacion de rankings -0,10.",
        "Los cuatro generadores de una corrida no son cuatro replicas estadisticas independientes.",
        "La tendencia por porcentaje esta corroborada, pero no probada con significacion robusta.",
    ])
    story += [callout(
        "Mensaje final: los sinteticos aportan algo de contexto historico; la sofisticacion del generador importa menos de lo esperado.",
        LIGHT_TEAL,
    ), Spacer(1, 8)]
    story += [subsection("Entregables derivados")]
    story += bullets([
        "Tablas finales de MAE, mejora porcentual y precision direccional.",
        "Grafico de cantidad sintetica frente a rendimiento.",
        "Comparacion calidad generativa frente a utilidad predictiva.",
        "Resumen listo para README y presentacion.",
    ])
    return story


GUIDES = [
    ("00_descarga_y_datos.pdf", build_00),
    ("01_eda_intradia.pdf", build_01),
    ("02_modelos_generativos.pdf", build_02),
    ("03_backfill_condicional.pdf", build_03),
    ("04_entrenamiento_predictor.pdf", build_04),
    ("05_analisis_resultados.pdf", build_05),
]


for filename, builder in GUIDES:
    doc = GuideDoc(str(OUT / filename), title=filename.replace("_", " ").replace(".pdf", ""),
                   author="Codex - guia del proyecto Taller B5-T1")
    doc.build(builder())
    print(OUT / filename)
