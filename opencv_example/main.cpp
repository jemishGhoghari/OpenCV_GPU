#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <regex>

using namespace cv;
using namespace std;

// ======================= Config =======================
struct YoloConfig
{
    string onnxPath;     // required
    string source = "0"; // camera index or video path
    string yamlPath;     // optional
    string namesPath;    // optional
    string savePath;     // optional video output
    int inputW = 640;
    int inputH = 640;
    float confThr = 0.25f;
    float iouThr = 0.45f;
    bool useCUDA = false;
};

// ======================= Utils ========================
static inline string trim(const string &s)
{
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == string::npos)
        return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

static vector<string> loadNamesFile(const string &path)
{
    vector<string> names;
    if (path.empty())
        return names;
    ifstream ifs(path);
    string line;
    while (getline(ifs, line))
    {
        line = trim(line);
        if (line.empty() || line[0] == '#')
            continue;
        names.push_back(line);
    }
    return names;
}

// Parse Ultralytics-style coco.yaml (names list/map)
static vector<string> parseCocoYaml(const string &path)
{
    if (path.empty())
        return {};
    ifstream ifs(path);
    if (!ifs.is_open())
        throw runtime_error("Could not open yaml: " + path);
    vector<string> names;
    string all;
    {
        ostringstream oss;
        oss << ifs.rdbuf();
        all = oss.str();
    }

    // Inline list: names: [a, b, c]
    {
        regex rx(R"(names\s*:\s*\[([^\]]*)\])");
        smatch m;
        if (regex_search(all, m, rx))
        {
            string inside = m[1].str(), cur;
            for (char c : inside)
            {
                if (c == ',')
                {
                    cur = trim(cur);
                    if (!cur.empty())
                        names.push_back(cur);
                    cur.clear();
                }
                else
                    cur.push_back(c);
            }
            cur = trim(cur);
            if (!cur.empty())
                names.push_back(cur);
            for (auto &s : names)
            {
                if (!s.empty() && (s.front() == '\'' || s.front() == '"'))
                    s.erase(s.begin());
                if (!s.empty() && (s.back() == '\'' || s.back() == '"'))
                    s.pop_back();
            }
            if (!names.empty())
                return names;
        }
    }
    // Inline map: names: {0: a, 1: b}
    {
        regex rx(R"(names\s*:\s*\{([^\}]*)\})");
        smatch m;
        if (regex_search(all, m, rx))
        {
            string inside = m[1].str();
            size_t start = 0;
            while (true)
            {
                size_t pos = inside.find(',', start);
                string tok = trim(pos == string::npos ? inside.substr(start) : inside.substr(start, pos - start));
                if (!tok.empty())
                {
                    size_t colon = tok.find(':');
                    if (colon != string::npos)
                    {
                        int idx = stoi(trim(tok.substr(0, colon)));
                        string val = trim(tok.substr(colon + 1));
                        if (!val.empty() && (val.front() == '\'' || val.front() == '"'))
                            val.erase(val.begin());
                        if (!val.empty() && (val.back() == '\'' || val.back() == '"'))
                            val.pop_back();
                        if ((int)names.size() <= idx)
                            names.resize(idx + 1);
                        names[idx] = val;
                    }
                }
                if (pos == string::npos)
                    break;
                start = pos + 1;
            }
            if (!names.empty())
                return names;
        }
    }
    // Multiline after "names:"
    {
        istringstream iss(all);
        string l;
        bool inNames = false;
        regex rx_idx(R"(^\s*([0-9]+)\s*:\s*(.+)\s*$)");
        regex rx_li(R"(^\s*-\s*(.+)\s*$)");
        while (getline(iss, l))
        {
            string t = trim(l);
            if (!inNames)
            {
                if (t.rfind("names:", 0) == 0)
                {
                    inNames = true;
                    continue;
                }
            }
            else
            {
                if (t.empty() || t[0] == '#')
                    continue;
                if (!isspace(l[0]) && t.find(':') != string::npos && t.rfind("-", 0) != 0)
                    break; // next top-level
                smatch m;
                if (regex_search(t, m, rx_idx))
                {
                    int idx = stoi(m[1].str());
                    string val = trim(m[2].str());
                    if (!val.empty() && (val.front() == '\'' || val.front() == '"'))
                        val.erase(val.begin());
                    if (!val.empty() && (val.back() == '\'' || val.back() == '"'))
                        val.pop_back();
                    if ((int)names.size() <= idx)
                        names.resize(idx + 1);
                    names[idx] = val;
                }
                else if (regex_search(t, m, rx_li))
                {
                    string val = trim(m[1].str());
                    if (!val.empty() && (val.front() == '\'' || val.front() == '"'))
                        val.erase(val.begin());
                    if (!val.empty() && (val.back() == '\'' || val.back() == '"'))
                        val.pop_back();
                    names.push_back(val);
                }
            }
        }
    }
    return names;
}

static vector<string> loadClassList(const YoloConfig &cfg)
{
    vector<string> names;
    if (!cfg.yamlPath.empty())
    {
        std::cout << "Loading class names from YAML file: " << cfg.yamlPath << "\n";
        names = parseCocoYaml(cfg.yamlPath);
    }
    else if (!cfg.namesPath.empty())
    {
        std::cout << "Loading class names from text file: " << cfg.namesPath << "\n";
        names = loadNamesFile(cfg.namesPath);
    }
    if (names.empty())
    {
        std::cout << "Warning: No class names found, using default names\n";
        names.resize(80);
        for (int i = 0; i < 80; ++i)
            names[i] = "cls_" + to_string(i);
    }
    std::cout << "Using " << names.size() << " classes\n";
    return names;
}

static Scalar classColor(int cid)
{
    uint32_t h = (uint32_t)cid * 2654435761u;
    return Scalar(h & 255, (h >> 8) & 255, (h >> 16) & 255);
}

// Letterbox to (inputW,inputH)
static Mat letterbox(const Mat &img, int newW, int newH, Vec4i &pad, float &scale)
{
    int w = img.cols, h = img.rows;
    float r = min((float)newW / w, (float)newH / h);
    int nw = int(round(w * r)), nh = int(round(h * r));
    scale = r;
    int left = (newW - nw) / 2, top = (newH - nh) / 2;

    Mat resized;
    resize(img, resized, Size(nw, nh));
    Mat out(newH, newW, img.type(), Scalar(114, 114, 114));
    resized.copyTo(out(Rect(left, top, nw, nh)));

    pad = Vec4i(left, top, newW - nw - left, newH - nh - top);
    return out;
}

static void drawDet(Mat &frame, const Rect &box, const string &label, float conf, const Scalar &color)
{
    rectangle(frame, box, color, 2);
    string text = format("%s %.2f", label.c_str(), conf);
    int base;
    Size t = getTextSize(text, FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
    int x = max(box.x, 0), y = max(box.y - t.height - 4, 0);
    rectangle(frame, Rect(Point(x, y), Size(t.width + 6, t.height + base + 6)), color, FILLED);
    putText(frame, text, Point(x + 3, y + t.height + 1), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
}

// ---------------------- FIXED PARSER ----------------------
// Accepts out with shape (1, C, N) or (1, N, C) and returns det as (N × C)
// Also detects [obj] presence and normalized coords.
static Mat makeDetRowsNxC(const Mat &out3d, int &C, int &N)
{
    CV_Assert(out3d.dims == 3);
    int d0 = out3d.size[0], d1 = out3d.size[1], d2 = out3d.size[2];
    CV_Assert(d0 == 1);

    // Guess which is C (channels/features) and which is N (preds)
    // C is usually small-ish (~ 4 + [obj?] + num_classes), e.g. 84 or 85 or up to ~300.
    // N is large (e.g. 8400).
    bool d1_is_C = (d1 <= 512);
    bool d2_is_C = (d2 <= 512);

    if (d1_is_C && !d2_is_C)
    { // (1, C, N)
        C = d1;
        N = d2;
        // reshape to (C × N), then transpose to (N × C)
        Mat tmp = out3d.reshape(1, C); // C × N
        return tmp.t();                // N × C
    }
    else if (!d1_is_C && d2_is_C)
    { // (1, N, C)
        C = d2;
        N = d1;
        // reshape directly to (N × C)
        return out3d.reshape(1, N); // N × C
    }
    else
    {
        // Ambiguous; fallback: treat the smaller as C
        if (d1 <= d2)
        {
            C = d1;
            N = d2;
            Mat tmp = out3d.reshape(1, C);
            return tmp.t();
        }
        else
        {
            C = d2;
            N = d1;
            return out3d.reshape(1, N);
        }
    }
}

// Robust decode with normalization + cxcywh/xyxy fallback + optional obj column
static void parseDetectionsRobust(
    const cv::Mat &out, float confThr,
    std::vector<cv::Rect> &boxes, std::vector<float> &scores, std::vector<int> &classIds,
    int imgW, int imgH, float scale, const cv::Vec4i &pad, int inputW, int inputH,
    int expectedNumClasses, bool debug = false)
{
    boxes.clear();
    scores.clear();
    classIds.clear();

    Mat det;
    int C = 0, N = 0;
    if (out.dims == 3)
        det = makeDetRowsNxC(out, C, N);
    else if (out.dims == 2)
    {
        det = out;
        C = det.cols;
        N = det.rows;
    }
    else
    {
        if (debug)
            cerr << "[parse] Unexpected dims: " << out.dims << "\n";
        return;
    }

    if (debug)
        cerr << "[parse] Using (N×C) = (" << N << "×" << C << ")\n";
    if (C < 6 || N <= 0)
    {
        if (debug)
            cerr << "[parse] C<6 or N<=0\n";
        return;
    }

    // Detect whether an objectness column exists:
    //   Case A: 4 + 1 + nc == C  -> has obj at index 4
    //   Case B: 4 + nc == C      -> obj is implicitly 1
    bool hasObj = false;
    int nc_guess_A = C - 5;
    int nc_guess_B = C - 4;
    if (nc_guess_A == expectedNumClasses)
        hasObj = true;
    else if (nc_guess_B == expectedNumClasses)
        hasObj = false;
    else
    {
        // If class file unknown/placeholder, heuristics:
        // If C==84 (typical 4+1+80) -> has obj
        // If C==80+4 (i.e., 84) also -> has obj; (80+4==84) same as above
        hasObj = (C >= 80 + 5); // coarse heuristic
    }
    if (debug)
        cerr << "[parse] hasObj=" << hasObj << " C=" << C << " nc_guessA=" << nc_guess_A << " nc_guessB=" << nc_guess_B << "\n";

    auto toBox = [&](float v0, float v1, float v2, float v3, bool cxcywh) -> cv::Rect2f
    {
        if (cxcywh)
            return Rect2f(v0 - v2 / 2.0f, v1 - v3 / 2.0f, v2, v3);
        else
            return Rect2f(v0, v1, v2 - v0, v3 - v1);
    };
    auto unletterbox = [&](cv::Rect2f r) -> cv::Rect2f
    {
        r.x -= pad[0];
        r.y -= pad[1];
        r.x /= scale;
        r.y /= scale;
        r.width /= scale;
        r.height /= scale;
        return r;
    };

    // Normalization guess (inspect the first row)
    const float *p0 = det.ptr<float>(0);
    int smallCnt = 0;
    for (int k = 0; k < 4; ++k)
        smallCnt += (p0[k] <= 1.5f);
    bool normalized = (smallCnt >= 3);
    if (debug)
        cerr << "[parse] normalized_guess=" << normalized << "\n";

    for (int i = 0; i < N; ++i)
    {
        const float *p = det.ptr<float>(i);

        float b0 = p[0], b1 = p[1], b2 = p[2], b3 = p[3];
        if (normalized)
        {
            b0 *= inputW;
            b1 *= inputH;
            b2 *= inputW;
            b3 *= inputH;
        }

        float obj = hasObj ? p[4] : 1.0f;
        int clsStart = hasObj ? 5 : 4;
        int cls = -1;
        float clsScore = 0.f;
        for (int c = clsStart; c < C; ++c)
            if (p[c] > clsScore)
            {
                clsScore = p[c];
                cls = c - clsStart;
            }
        float conf = obj * clsScore;
        if (conf < confThr)
            continue;

        Rect2f r_cx = toBox(b0, b1, b2, b3, /*cxcywh=*/true);
        Rect2f r_xy = toBox(b0, b1, b2, b3, /*cxcywh=*/false);

        auto valid = [&](const Rect2f &r)
        {
            return r.width > 1.f && r.height > 1.f &&
                   r.x + r.width > 1.f && r.y + r.height > 1.f;
        };
        Rect2f r = valid(r_cx) ? r_cx : r_xy;

        r = unletterbox(r);

        int left = (int)std::round(r.x);
        int top = (int)std::round(r.y);
        int width = (int)std::round(r.width);
        int height = (int)std::round(r.height);
        if (width <= 1 || height <= 1)
            continue;

        Rect box(left, top, width, height);
        box.x = max(0, min(box.x, imgW - 1));
        box.y = max(0, min(box.y, imgH - 1));
        box.width = max(0, min(box.width, imgW - box.x));
        box.height = max(0, min(box.height, imgH - box.y));
        if (box.area() <= 0)
            continue;

        boxes.push_back(box);
        scores.push_back(conf);
        classIds.push_back(cls);
    }
}

static void printHelp(const char *prog)
{
    cout << "Usage:\n"
            "  "
         << prog << " <yolo11.onnx> [source]\n"
                    "Options:\n"
                    "  --yaml path        Load classes from coco.yaml\n"
                    "  --names path       Load classes from .names file\n"
                    "  --conf f           Confidence threshold (default 0.25)\n"
                    "  --iou f            IoU threshold for NMS (default 0.45)\n"
                    "  --size WxH         Inference size (default 640x640)\n"
                    "  --save out.mp4     Save annotated video\n"
                    "  --cuda             Use CUDA DNN backend (if available)\n";
}

static bool parseSize(const string &s, int &w, int &h)
{
    size_t x = s.find('x');
    if (x == string::npos)
        return false;
    try
    {
        w = stoi(s.substr(0, x));
        h = stoi(s.substr(x + 1));
        return (w > 0 && h > 0);
    }
    catch (...)
    {
        return false;
    }
}

// ======================= Main =========================
int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printHelp(argv[0]);
        return 0;
    }

    YoloConfig cfg;
    cfg.onnxPath = argv[1];
    if (argc >= 3 && argv[2][0] != '-')
        cfg.source = argv[2];

    // parse flags
    for (int i = 2; i < argc; ++i)
    {
        string a = argv[i];
        if (a == "--yaml" && i + 1 < argc)
            cfg.yamlPath = argv[++i];
        else if (a == "--names" && i + 1 < argc)
            cfg.namesPath = argv[++i];
        else if (a == "--conf" && i + 1 < argc)
            cfg.confThr = stof(argv[++i]);
        else if (a == "--iou" && i + 1 < argc)
            cfg.iouThr = stof(argv[++i]);
        else if (a == "--size" && i + 1 < argc)
        {
            if (!parseSize(argv[++i], cfg.inputW, cfg.inputH))
            {
                cerr << "Bad --size\n";
                return 1;
            }
        }
        else if (a == "--save" && i + 1 < argc)
            cfg.savePath = argv[++i];
        else if (a == "--cuda")
            cfg.useCUDA = true;
        else if (a.rfind("--", 0) == 0)
        {
            cerr << "Unknown option: " << a << "\n";
            return 1;
        }
    }

    // Load classes
    vector<string> classNames;
    try
    {
        classNames = loadClassList(cfg);
    }
    catch (const exception &e)
    {
        cerr << e.what() << "\n";
        return 2;
    }
    cout << "Loaded " << classNames.size() << " classes\n";

    // Open source
    VideoCapture cap;
    if (cfg.source.size() == 1 && isdigit(cfg.source[0]))
        cap.open(stoi(cfg.source), CAP_V4L2);
    else
        cap.open(cfg.source, CAP_V4L2);
    if (!cap.isOpened())
    {
        cerr << "ERROR: cannot open source: " << cfg.source << "\n";
        return 3;
    }
    cap.set(CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));

    // Load net
    dnn::Net net = dnn::readNet(cfg.onnxPath);
    if (net.empty())
    {
        cerr << "ERROR: failed to load ONNX: " << cfg.onnxPath << "\n";
        return 4;
    }

    try
    {
        if (cfg.useCUDA)
        {
#ifdef HAVE_CUDA
            net.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
            net.setPreferableTarget(dnn::DNN_TARGET_CUDA_FP16);
#else
            cerr << "CUDA not available in this OpenCV build. Falling back to CPU.\n";
            net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(dnn::DNN_TARGET_CPU);
#endif
        }
        else
        {
            net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(dnn::DNN_TARGET_CPU);
        }
    }
    catch (const cv::Exception &e)
    {
        cerr << "DNN backend/target set failed: " << e.what() << "\n";
    }

    // Warmup
    {
        Mat dummy(cfg.inputH, cfg.inputW, CV_8UC3, Scalar(114, 114, 114));
        Mat blob = dnn::blobFromImage(dummy, 1.0 / 255.0, Size(cfg.inputW, cfg.inputH), Scalar(), true, false);
        net.setInput(blob);
        (void)net.forward();
    }

    // Optional writer
    VideoWriter writer;
    bool save = false;
    if (!cfg.savePath.empty())
    {
        int fourcc = VideoWriter::fourcc('m', 'p', '4', 'v');
        double fps = cap.get(CAP_PROP_FPS);
        if (fps <= 0)
            fps = 25.0;
        Mat probe;
        cap >> probe;
        if (probe.empty())
        {
            cerr << "ERROR: empty first frame\n";
            return 5;
        }
        writer.open(cfg.savePath, fourcc, fps, probe.size(), true);
        if (!writer.isOpened())
        {
            cerr << "ERROR: cannot open writer: " << cfg.savePath << "\n";
            return 6;
        }
        cap.set(CAP_PROP_POS_FRAMES, 0);
        save = true;
        cout << "Saving to: " << cfg.savePath << "\n";
    }

    cout << "Running. Press 'q' or ESC to quit.\n";
    TickMeter tm;
    static bool printedShape = false;

    // For expected class count detection in parser
    const int expectedNumClasses = (int)classNames.size();

    for (;;)
    {
        Mat frame;
        cap >> frame;
        if (frame.empty())
            break;

        Vec4i pad;
        float scale = 1.f;
        Mat in = letterbox(frame, cfg.inputW, cfg.inputH, pad, scale);

        Mat blob = dnn::blobFromImage(in, 1.0 / 255.0, Size(cfg.inputW, cfg.inputH),
                                      Scalar(), /*swapRB=*/true, /*crop=*/false);

        tm.start();
        net.setInput(blob);
        vector<Mat> outs;
        net.forward(outs);
        tm.stop();

        Mat out = outs.empty() ? net.forward() : outs[0];

        if (!printedShape)
        {
            cerr << "[DNN] out.dims=" << out.dims << " sizes=";
            for (int i = 0; i < out.dims; ++i)
                cerr << out.size[i] << " ";
            cerr << " type=" << out.type() << " (CV_32F is 5)\n";
            printedShape = true;
        }

        vector<Rect> boxes;
        vector<float> scores;
        vector<int> classIds;
        parseDetectionsRobust(out, cfg.confThr, boxes, scores, classIds,
                              frame.cols, frame.rows, scale, pad, cfg.inputW, cfg.inputH,
                              expectedNumClasses, /*debug=*/false);

        vector<int> keep;
        dnn::NMSBoxes(boxes, scores, cfg.confThr, cfg.iouThr, keep);

        for (int i : keep)
        {
            if (i < 0 || i >= (int)boxes.size())
                continue;
            int cid = (i < (int)classIds.size()) ? classIds[i] : 0;
            string label = (cid >= 0 && cid < (int)classNames.size()) ? classNames[cid] : ("id_" + to_string(cid));
            drawDet(frame, boxes[i], label, scores[i], classColor(cid));
        }

        // if (!keep.empty() && !classIds.empty()) {
        //     int idx = keep[0];
        //     if (idx >= 0 && idx < (int)classIds.size()) {
        //         int cid = classIds[idx];
        //         if (cid >= 0 && cid < (int)classNames.size())
        //             cout << "Detected Class: " << classNames[cid] << "\n";
        //     }
        // }

        double fps = 1e3 / tm.getTimeMilli();
        tm.reset();
        putText(frame, format("FPS: %.1f", fps), Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0, 0, 0), 3);
        putText(frame, format("FPS: %.1f", fps), Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(255, 255, 255), 1);

        imshow("YOLOv11 - OpenCV DNN (fixed)", frame);
        if (save)
            writer << frame;

        int key = waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q')
            break;
    }
    return 0;
}
