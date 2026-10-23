#include "application.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <string>
#include <algorithm>

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <imgui.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace application {

// ===========================================================================
// Математика (Column-major: m[col][row], стандарт GLSL и Vulkan)
// ===========================================================================

struct Vec3 {
    float x = 0.f, y = 0.f, z = 0.f;

    Vec3 operator+(const Vec3& o) const { return { x + o.x, y + o.y, z + o.z }; }
    Vec3 operator-(const Vec3& o) const { return { x - o.x, y - o.y, z - o.z }; }
    Vec3 operator*(float s) const { return { x * s, y * s, z * s }; }
};

static inline float vec3_dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline float vec3_length(const Vec3& v) {
    return std::sqrt(vec3_dot(v, v));
}

static inline Vec3 vec3_normalize(const Vec3& v) {
    float l = vec3_length(v);
    if (l > 1e-6f) return { v.x / l, v.y / l, v.z / l };
    return { 0.f, 0.f, 1.f };
}

static inline Vec3 vec3_cross(const Vec3& a, const Vec3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

struct Mat4 {
    float m[4][4]{};

    // Единичная матрица
    static Mat4 identity() {
        Mat4 r;
        r.m[0][0] = r.m[1][1] = r.m[2][2] = r.m[3][3] = 1.f;
        return r;
    }
};

// Умножение матриц: result = a * b (по столбцам)
static Mat4 mat4_mul(const Mat4& a, const Mat4& b) {
    Mat4 r;
    for (int col = 0; col < 4; ++col)
        for (int row = 0; row < 4; ++row)
            for (int k = 0; k < 4; ++k)
                r.m[col][row] += a.m[k][row] * b.m[col][k];
    return r;
}

// Матрица переноса (сдвига)
static Mat4 mat4_translate(float tx, float ty, float tz) {
    Mat4 r = Mat4::identity();
    r.m[3][0] = tx; r.m[3][1] = ty; r.m[3][2] = tz;
    return r;
}

// Матрица масштабирования
static Mat4 mat4_scale(float sx, float sy, float sz) {
    Mat4 r = Mat4::identity();
    r.m[0][0] = sx; r.m[1][1] = sy; r.m[2][2] = sz;
    return r;
}

// Вращение вокруг оси X
static Mat4 mat4_rotate_x(float a) {
    Mat4 r = Mat4::identity();
    r.m[1][1] =  std::cos(a); r.m[1][2] =  std::sin(a);
    r.m[2][1] = -std::sin(a); r.m[2][2] =  std::cos(a);
    return r;
}

// Вращение вокруг оси Y
static Mat4 mat4_rotate_y(float a) {
    Mat4 r = Mat4::identity();
    r.m[0][0] =  std::cos(a); r.m[0][2] = -std::sin(a);
    r.m[2][0] =  std::sin(a); r.m[2][2] =  std::cos(a);
    return r;
}

// Вращение вокруг оси Z
static Mat4 mat4_rotate_z(float a) {
    Mat4 r = Mat4::identity();
    r.m[0][0] =  std::cos(a); r.m[0][1] =  std::sin(a);
    r.m[1][0] = -std::sin(a); r.m[1][1] =  std::cos(a);
    return r;
}

// Перспективная проекция для Vulkan (диапазон глубины Z [0, 1], инверсия оси Y)
static Mat4 mat4_perspective(float fovY_rad, float aspect, float near_z, float far_z) {
    float f = 1.f / std::tan(fovY_rad * 0.5f);
    Mat4 r;
    r.m[0][0] = f / aspect;
    r.m[1][1] = -f;                                         // Инверсия Y для Vulkan
    r.m[2][2] = far_z / (near_z - far_z);
    r.m[3][2] = (near_z * far_z) / (near_z - far_z);
    r.m[2][3] = -1.f;
    return r;
}

// Ортографическая проекция для Vulkan
static Mat4 mat4_ortho(float left, float right, float bottom, float top,
                       float near_z, float far_z) {
    Mat4 r;
    r.m[0][0] =  2.f / (right - left);
    r.m[1][1] = -2.f / (top - bottom);                      // Инверсия Y для Vulkan
    r.m[2][2] =  1.f / (near_z - far_z);
    r.m[3][0] = -(right + left)   / (right - left);
    r.m[3][1] =  (top   + bottom) / (top   - bottom);
    r.m[3][2] =  near_z / (near_z - far_z);
    r.m[3][3] =  1.f;
    return r;
}

// Матрица вида (LookAt) для камеры
static Mat4 mat4_look_at(Vec3 eye, Vec3 center, Vec3 up) {
    Vec3 f = vec3_normalize(center - eye);
    Vec3 rr = vec3_normalize(vec3_cross(f, up));
    Vec3 u = vec3_cross(rr, f);

    Mat4 mat = Mat4::identity();
    mat.m[0][0] = rr.x; mat.m[1][0] = rr.y; mat.m[2][0] = rr.z; mat.m[3][0] = -vec3_dot(rr, eye);
    mat.m[0][1] = u.x;  mat.m[1][1] = u.y;  mat.m[2][1] = u.z;  mat.m[3][1] = -vec3_dot(u, eye);
    mat.m[0][2] = -f.x; mat.m[1][2] = -f.y; mat.m[2][2] = -f.z; mat.m[3][2] =  vec3_dot(f, eye);
    mat.m[3][3] = 1.f;
    return mat;
}

// Вычисление обратной матрицы 4х4
static bool mat4_inverse(const Mat4& in, Mat4& out) {
    float m[16];
    for (int col = 0; col < 4; ++col)
        for (int row = 0; row < 4; ++row)
            m[col * 4 + row] = in.m[col][row];

    float inv[16];
    inv[0] = m[5]  * m[10] * m[15] - m[5]  * m[11] * m[14] - m[9]  * m[6]  * m[15] + 
             m[9]  * m[7]  * m[14] + m[13] * m[6]  * m[11] - m[13] * m[7]  * m[10];
    inv[4] = -m[4]  * m[10] * m[15] + m[4]  * m[11] * m[14] + m[8]  * m[6]  * m[15] - 
              m[8]  * m[7]  * m[14] - m[12] * m[6]  * m[11] + m[12] * m[7]  * m[10];
    inv[8] = m[4]  * m[9] * m[15] - m[4]  * m[11] * m[13] - m[8]  * m[5] * m[15] + 
             m[8]  * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4]  * m[9] * m[14] + m[4]  * m[10] * m[13] + m[8]  * m[5] * m[14] - 
               m[8]  * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1]  * m[10] * m[15] + m[1]  * m[11] * m[14] + m[9]  * m[2] * m[15] - 
              m[9]  * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0]  * m[10] * m[15] - m[0]  * m[11] * m[14] - m[8]  * m[2] * m[15] + 
             m[8]  * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0]  * m[9] * m[15] + m[0]  * m[11] * m[13] + m[8]  * m[1] * m[15] - 
              m[8]  * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0]  * m[9] * m[14] - m[0]  * m[10] * m[13] - m[8]  * m[1] * m[14] + 
              m[8]  * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1]  * m[6] * m[15] - m[1]  * m[7] * m[14] - m[5]  * m[2] * m[15] + 
             m[5]  * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0]  * m[6] * m[15] + m[0]  * m[7] * m[14] + m[4]  * m[2] * m[15] - 
              m[4]  * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0]  * m[5] * m[15] - m[0]  * m[7] * m[13] - m[4]  * m[1] * m[15] + 
              m[4]  * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0]  * m[5] * m[14] + m[0]  * m[6] * m[13] + m[4]  * m[1] * m[14] - 
               m[4]  * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - 
              m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + 
             m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - 
               m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + 
              m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

    float det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    if (std::abs(det) < 1e-8f) {
        out = Mat4::identity();
        return false;
    }

    float invDet = 1.0f / det;
    for (int col = 0; col < 4; ++col)
        for (int row = 0; row < 4; ++row)
            out.m[col][row] = inv[col * 4 + row] * invDet;

    return true;
}

// Транспонирование матрицы
static Mat4 mat4_transpose(const Mat4& a) {
    Mat4 r;
    for (int col = 0; col < 4; ++col)
        for (int row = 0; row < 4; ++row)
            r.m[col][row] = a.m[row][col];
    return r;
}

// Вычисление матрицы нормалей: transpose(inverse(model))
static Mat4 mat4_normal_matrix(const Mat4& model) {
    Mat4 inv;
    if (!mat4_inverse(model, inv)) return Mat4::identity();
    return mat4_transpose(inv);
}

// ===========================================================================
// Структуры геометрии и генераторы мешей
// ===========================================================================

// Вершина с атрибутами позиции, нормали и цвета
struct Vertex {
    float pos[3];    // X, Y, Z
    float normal[3]; // Nx, Ny, Nz (процедурная нормаль)
    float color[3];  // R, G, B
};

struct MeshData {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
};

// Преобразование HSV в RGB для процедурной раскраски
static void hsv_to_rgb(float h, float& r, float& g, float& b) {
    float h6 = h * 6.f;
    int   hi = (int)h6 % 6;
    float f  = h6 - (float)(int)h6;
    switch (hi) {
        case 0: r=1.f;   g=f;    b=0.f;   break;
        case 1: r=1.f-f; g=1.f;  b=0.f;   break;
        case 2: r=0.f;   g=1.f;  b=f;     break;
        case 3: r=0.f;   g=1.f-f;b=1.f;   break;
        case 4: r=f;     g=0.f;  b=1.f;   break;
        default:r=1.f;   g=0.f;  b=1.f-f; break;
    }
}

// Процедурная генерация Тора (Вариант 7) с аналитическими нормалями
static MeshData buildTorus(float R, float r, int segments, int sides) {
    MeshData md;
    for (int i = 0; i <= segments; ++i) {
        float theta = 2.f * (float)M_PI * i / segments;
        float ct = std::cos(theta), st = std::sin(theta);

        // Радужный спектральный цвет вдоль большого кольца
        float cr, cg, cb;
        hsv_to_rgb(theta / (2.f * (float)M_PI), cr, cg, cb);

        for (int j = 0; j <= sides; ++j) {
            float phi = 2.f * (float)M_PI * j / sides;
            float cp = std::cos(phi), sp = std::sin(phi);

            Vertex v;
            // Координаты вершины тора
            v.pos[0]    = (R + r * cp) * ct;
            v.pos[1]    = (R + r * cp) * st;
            v.pos[2]    = r * sp;
            // Процедурно вычисленная аналитическая нормаль тора
            v.normal[0] = cp * ct;
            v.normal[1] = cp * st;
            v.normal[2] = sp;
            // Процедурный цвет
            v.color[0]  = cr;
            v.color[1]  = cg;
            v.color[2]  = cb;
            md.vertices.push_back(v);
        }
    }

    // Соединение вершин в треугольники
    for (int i = 0; i < segments; ++i) {
        for (int j = 0; j < sides; ++j) {
            uint32_t a = i * (sides + 1) + j;
            uint32_t b = a + 1;
            uint32_t c = (i + 1) * (sides + 1) + j;
            uint32_t d = c + 1;
            md.indices.push_back(a); md.indices.push_back(c); md.indices.push_back(b);
            md.indices.push_back(b); md.indices.push_back(c); md.indices.push_back(d);
        }
    }
    return md;
}

// Процедурный пол (плоскость) для наглядной демонстрации света и теней
static MeshData buildFloor(float size, int divisions, float zPos) {
    MeshData md;
    float step = (size * 2.f) / divisions;

    for (int i = 0; i <= divisions; ++i) {
        float y = -size + i * step;
        for (int j = 0; j <= divisions; ++j) {
            float x = -size + j * step;
            Vertex v;
            v.pos[0] = x;
            v.pos[1] = y;
            v.pos[2] = zPos;
            // Нормаль пола направлена строго вверх
            v.normal[0] = 0.f;
            v.normal[1] = 0.f;
            v.normal[2] = 1.f;

            // Шахматный легкий узор в цвете вершин
            bool isDark = ((i + j) % 2) == 0;
            float val = isDark ? 0.85f : 1.0f;
            v.color[0] = val;
            v.color[1] = val;
            v.color[2] = val;
            md.vertices.push_back(v);
        }
    }

    for (int i = 0; i < divisions; ++i) {
        for (int j = 0; j < divisions; ++j) {
            uint32_t a = i * (divisions + 1) + j;
            uint32_t b = a + 1;
            uint32_t c = (i + 1) * (divisions + 1) + j;
            uint32_t d = c + 1;
            md.indices.push_back(a); md.indices.push_back(c); md.indices.push_back(b);
            md.indices.push_back(b); md.indices.push_back(c); md.indices.push_back(d);
        }
    }
    return md;
}

// Процедурная сфера для визуализации источников света (маркеров лампочек)
static MeshData buildSphere(float radius, int rings, int sectors) {
    MeshData md;
    for (int r = 0; r <= rings; ++r) {
        float phi = (float)M_PI * r / rings;
        float sp = std::sin(phi), cp = std::cos(phi);

        for (int s = 0; s <= sectors; ++s) {
            float theta = 2.f * (float)M_PI * s / sectors;
            float st = std::sin(theta), ct = std::cos(theta);

            Vertex v;
            v.normal[0] = sp * ct;
            v.normal[1] = sp * st;
            v.normal[2] = cp;
            v.pos[0] = v.normal[0] * radius;
            v.pos[1] = v.normal[1] * radius;
            v.pos[2] = v.normal[2] * radius;
            v.color[0] = 1.f;
            v.color[1] = 1.f;
            v.color[2] = 1.f;
            md.vertices.push_back(v);
        }
    }

    for (int r = 0; r < rings; ++r) {
        for (int s = 0; s < sectors; ++s) {
            uint32_t a = r * (sectors + 1) + s;
            uint32_t b = a + 1;
            uint32_t c = (r + 1) * (sectors + 1) + s;
            uint32_t d = c + 1;
            md.indices.push_back(a); md.indices.push_back(c); md.indices.push_back(b);
            md.indices.push_back(b); md.indices.push_back(c); md.indices.push_back(d);
        }
    }
    return md;
}

// ===========================================================================
// Структуры для передачи на GPU (соответствуют std140 / std430 в шейдерах)
// ===========================================================================

// Направленный свет
struct GpuDirectionalLight {
    float direction[4]; // xyz: направление, w: включен (1.0 или 0.0)
    float color[4];     // rgb: цвет, w: интенсивность
};

// UBO Сцены (Set 0, Binding 0)
struct GpuSceneUBO {
    Mat4 view;                  // Матрица камеры
    Mat4 proj;                  // Матрица проекции
    Mat4 viewProj;              // View * Proj
    float cameraPos[4];         // Положение камеры
    float ambientLight[4];      // rgb: цвет фона, w: интенсивность
    GpuDirectionalLight dirLight;// Направленный свет
    int32_t numPointLights;     // Количество точечных источников
    int32_t numSpotLights;      // Количество прожекторов
    int32_t pad0;               // Выравнивание std140
    int32_t pad1;               // Выравнивание std140
};
static_assert(sizeof(GpuSceneUBO) == 64 * 3 + 16 * 2 + 32 + 16, "SceneUBO size mismatch");

// Точечный свет для SSBO (Set 0, Binding 1)
struct GpuPointLight {
    float position[4]; // xyz: координаты, w: 1.0
    float color[4];    // rgb: цвет, w: интенсивность
    float params[4];   // x: const, y: lin, z: quad (затухание), w: включен
};
static_assert(sizeof(GpuPointLight) == 48, "PointLight size mismatch");

// Прожектор для SSBO (Set 0, Binding 2)
struct GpuSpotLight {
    float position[4];  // xyz: координаты, w: 1.0
    float direction[4]; // xyz: направление луча, w: 0.0
    float color[4];     // rgb: цвет, w: интенсивность
    float params[4];    // x: cos(внутр), y: cos(внеш), z: quad затухание, w: включен
};
static_assert(sizeof(GpuSpotLight) == 64, "SpotLight size mismatch");

// UBO Модели (Set 1, Binding 0)
struct GpuModelUBO {
    Mat4 model;        // Матрица модели
    Mat4 normalMatrix; // Матрица нормалей
    float albedo[4];   // rgb: альбедо, w: не используется
    float specular[4]; // rgb: цвет блика, w: показатель блеска (shininess)
    float emissive[4]; // rgb: цвет свечения, w: unlit-флаг (1.0 = без освещения)
};
static_assert(sizeof(GpuModelUBO) == 64 * 2 + 16 * 3, "ModelUBO size mismatch");

// ===========================================================================
// Структуры для управления источниками света в интерфейсе ImGUI
// ===========================================================================

// Точечный источник
struct PointLightItem {
    bool enabled = true;
    Vec3 position{ 2.0f, 0.0f, 1.2f };
    float color[3]{ 1.0f, 0.75f, 0.3f }; // Теплый янтарный цвет
    float intensity = 2.5f;
    float constAtt = 1.0f;
    float linAtt = 0.09f;
    float quadAtt = 0.032f; // Квадратичное затухание
    bool animated = true;
    float animSpeed = 1.0f;
    float animRadius = 2.2f;
    float animHeight = 1.2f;
    float animOffset = 0.0f;
};

// Прожектор
struct SpotLightItem {
    bool enabled = true;
    Vec3 position{ 0.0f, -2.5f, 2.5f };
    Vec3 target{ 0.0f, 0.0f, 0.0f };
    float color[3]{ 0.3f, 0.7f, 1.0f }; // Холодный голубой цвет
    float intensity = 4.0f;
    float innerAngleDeg = 15.0f; // Внутренний угол конуса
    float outerAngleDeg = 28.0f; // Внешний угол конуса
    float quadAtt = 0.04f;
    bool animated = false;
    float animSpeed = 0.8f;
    float animSweep = 1.5f;
};

// ===========================================================================
// Глобальное состояние приложения
// ===========================================================================

// Камера
static int   g_cam_mode      = 0; // 0: Орбитальная, 1: Ручная
static float g_cam_dist      = 5.5f;
static float g_cam_azimuth   = 0.45f;
static float g_cam_elevation = 0.40f;
static Vec3  g_cam_target    = { 0.f, 0.f, 0.f };
static Vec3  g_cam_pos       = { 0.f, -5.f, 2.5f };
static float g_cam_fov_deg   = 45.f;
static int   g_use_perspective = 1;
static float g_ortho_size    = 3.0f;

// Направленный и фоновый свет
static bool  g_dir_enabled   = true;
static float g_dir_pitch     = 0.85f;
static float g_dir_yaw       = 0.75f;
static float g_dir_color[3]  = { 1.0f, 0.98f, 0.92f };
static float g_dir_intensity = 0.9f;

static float g_ambient_color[3] = { 0.15f, 0.18f, 0.22f };
static float g_ambient_intensity = 0.5f;

// Динамические списки источников света
static std::vector<PointLightItem> g_point_lights;
static std::vector<SpotLightItem>  g_spot_lights;

// Материалы объектов
static float g_torus_albedo[3]   = { 0.95f, 0.65f, 0.25f };
static float g_torus_specular[3] = { 1.0f, 1.0f, 1.0f };
static float g_torus_shininess   = 48.0f;
static bool  g_torus_rainbow     = true;

static bool  g_floor_enabled     = true;
static float g_floor_albedo[3]   = { 0.35f, 0.38f, 0.42f };
static float g_floor_specular[3] = { 0.35f, 0.35f, 0.35f };
static float g_floor_shininess   = 16.0f;

// Трансформация тора
static float g_model_pos[3]   = { 0.f, 0.f, 0.f };
static float g_model_rot[3]   = { 0.f, 0.f, 0.f };
static float g_model_scale[3] = { 1.f, 1.f, 1.f };
static bool  g_model_spin     = true;

// Настройки визуализации источников
static bool  g_show_gizmos    = true;
static bool  g_show_cones     = true;

static double g_prev_time     = -1.0;
static float  g_total_time    = 0.0f;

// ===========================================================================
// Ресурсы Vulkan
// ===========================================================================

struct GpuMesh {
    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VmaAllocation vertexAlloc = VK_NULL_HANDLE;
    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VmaAllocation indexAlloc = VK_NULL_HANDLE;
    uint32_t indexCount = 0;
};

static GpuMesh g_mesh_torus;
static GpuMesh g_mesh_floor;
static GpuMesh g_mesh_sphere;

// Динамический вершинный буфер для отрисовки линий конусов прожекторов
static VkBuffer      g_line_vb     = VK_NULL_HANDLE;
static VmaAllocation g_line_alloc  = VK_NULL_HANDLE;
static void*         g_line_mapped = nullptr;
static const uint32_t MAX_LINE_VERTS = 4096;

// Буферы сцены (UBO и SSBO)
static VkBuffer      g_scene_ubo_buffer = VK_NULL_HANDLE;
static VmaAllocation g_scene_ubo_alloc  = VK_NULL_HANDLE;
static void*         g_scene_ubo_mapped = nullptr;

static VkBuffer      g_point_ssbo_buffer = VK_NULL_HANDLE;
static VmaAllocation g_point_ssbo_alloc  = VK_NULL_HANDLE;
static void*         g_point_ssbo_mapped = nullptr;
static const uint32_t MAX_POINT_LIGHTS = 64;

static VkBuffer      g_spot_ssbo_buffer = VK_NULL_HANDLE;
static VmaAllocation g_spot_ssbo_alloc  = VK_NULL_HANDLE;
static void*         g_spot_ssbo_mapped = nullptr;
static const uint32_t MAX_SPOT_LIGHTS = 64;

// Пул слотов для UBO моделей (Set 1)
struct ModelSlot {
    VkBuffer ubo = VK_NULL_HANDLE;
    VmaAllocation alloc = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDescriptorSet desc_set = VK_NULL_HANDLE;
};
static const uint32_t MAX_MODEL_SLOTS = 32;
static std::vector<ModelSlot> g_model_slots;

// Макеты дескрипторов и пайплайны
static VkDescriptorSetLayout g_scene_desc_layout = VK_NULL_HANDLE;
static VkDescriptorSetLayout g_model_desc_layout = VK_NULL_HANDLE;
static VkDescriptorPool      g_desc_pool         = VK_NULL_HANDLE;
static VkDescriptorSet       g_scene_desc_set    = VK_NULL_HANDLE;

static VkPipelineLayout      g_pipe_layout       = VK_NULL_HANDLE;
static VkPipeline            g_mesh_pipeline     = VK_NULL_HANDLE;
static VkPipeline            g_line_pipeline     = VK_NULL_HANDLE;

// ===========================================================================
// Вспомогательные функции
// ===========================================================================

// Чтение скомпилированного шейдера из файла
static std::vector<char> readFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw std::runtime_error("Cannot open: " + path);
    size_t sz = (size_t)file.tellg();
    std::vector<char> buf(sz);
    file.seekg(0);
    file.read(buf.data(), (std::streamsize)sz);
    return buf;
}

// Создание шейдерного модуля Vulkan
static VkShaderModule makeShader(const std::vector<char>& code) {
    auto& ctx = graphics::internal::context;
    VkShaderModuleCreateInfo ci{};
    ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = code.size();
    ci.pCode    = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule mod;
    if (vkCreateShaderModule(ctx.device, &ci, nullptr, &mod) != VK_SUCCESS)
        throw std::runtime_error("vkCreateShaderModule failed");
    return mod;
}

// Загрузка данных в GPU через промежуточный staging буфер
static bool upload(VkBuffer dst, const void* data, VkDeviceSize size) {
    auto& ctx = graphics::internal::context;

    VkBuffer staging; VmaAllocation sa;
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size  = size;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
              | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo ai;
    if (vmaCreateBuffer(ctx.allocator, &bci, &aci, &staging, &sa, &ai) != VK_SUCCESS)
        return false;
    memcpy(ai.pMappedData, data, size);

    VkCommandPool pool;
    VkCommandPoolCreateInfo cpi{};
    cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpi.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    cpi.queueFamilyIndex = ctx.graphics_queue_index;
    vkCreateCommandPool(ctx.device, &cpi, nullptr, &pool);

    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = pool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    vkAllocateCommandBuffers(ctx.device, &cbai, &cmd);

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);
    VkBufferCopy copy{ 0, 0, size };
    vkCmdCopyBuffer(cmd, staging, dst, 1, &copy);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1; submit.pCommandBuffers = &cmd;
    vkQueueSubmit(ctx.graphics_queue, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx.graphics_queue);

    vkDestroyCommandPool(ctx.device, pool, nullptr);
    vmaDestroyBuffer(ctx.allocator, staging, sa);
    return true;
}

// Создание буфера в быстрой памяти GPU (Device Local)
static bool makeDevBuf(VkDeviceSize size, VkBufferUsageFlags usage,
                       VkBuffer& buf, VmaAllocation& alloc) {
    auto& ctx = graphics::internal::context;
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size; bci.usage = usage; bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    return vmaCreateBuffer(ctx.allocator, &bci, &aci, &buf, &alloc, nullptr) == VK_SUCCESS;
}

// Загрузка вершинного и индексного буфера меша
static bool createGpuMesh(const MeshData& md, GpuMesh& outMesh) {
    VkDeviceSize vb_sz = md.vertices.size() * sizeof(Vertex);
    if (!makeDevBuf(vb_sz, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    outMesh.vertexBuffer, outMesh.vertexAlloc) ||
        !upload(outMesh.vertexBuffer, md.vertices.data(), vb_sz)) {
        return false;
    }

    VkDeviceSize ib_sz = md.indices.size() * sizeof(uint32_t);
    if (!makeDevBuf(ib_sz, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    outMesh.indexBuffer, outMesh.indexAlloc) ||
        !upload(outMesh.indexBuffer, md.indices.data(), ib_sz)) {
        return false;
    }

    outMesh.indexCount = (uint32_t)md.indices.size();
    return true;
}

static void destroyGpuMesh(GpuMesh& mesh) {
    auto& ctx = graphics::internal::context;
    if (mesh.indexBuffer)  { vmaDestroyBuffer(ctx.allocator, mesh.indexBuffer, mesh.indexAlloc); mesh.indexBuffer = VK_NULL_HANDLE; }
    if (mesh.vertexBuffer) { vmaDestroyBuffer(ctx.allocator, mesh.vertexBuffer, mesh.vertexAlloc); mesh.vertexBuffer = VK_NULL_HANDLE; }
}

// ===========================================================================
// Инициализация приложения и Vulkan ресурсов
// ===========================================================================

bool initialize() {
    auto& ctx = graphics::internal::context;

    // 1. Создаем начальные источники света в сцене
    g_point_lights.clear();
    {
        PointLightItem pl1;
        pl1.position = { 2.0f, 0.0f, 1.2f };
        pl1.color[0] = 1.0f; pl1.color[1] = 0.6f; pl1.color[2] = 0.2f; // Теплый золотой
        pl1.intensity = 3.5f;
        pl1.animated = true;
        pl1.animRadius = 2.2f;
        pl1.animOffset = 0.0f;
        g_point_lights.push_back(pl1);

        PointLightItem pl2;
        pl2.position = { -2.0f, 0.0f, 1.2f };
        pl2.color[0] = 0.2f; pl2.color[1] = 0.7f; pl2.color[2] = 1.0f; // Холодный голубой
        pl2.intensity = 3.5f;
        pl2.animated = true;
        pl2.animRadius = 2.2f;
        pl2.animOffset = (float)M_PI;
        g_point_lights.push_back(pl2);
    }

    g_spot_lights.clear();
    {
        SpotLightItem sl;
        sl.position = { 0.0f, -2.6f, 2.5f };
        sl.target   = { 0.0f, 0.0f, 0.0f };
        sl.color[0] = 1.0f; sl.color[1] = 0.3f; sl.color[2] = 0.5f; // Пурпурный прожектор
        sl.intensity = 5.0f;
        sl.innerAngleDeg = 15.0f;
        sl.outerAngleDeg = 28.0f;
        sl.animated = false;
        g_spot_lights.push_back(sl);
    }

    // 2. Создаем меши (Тор, Пол, Сфера для маркеров света)
    if (!createGpuMesh(buildTorus(1.0f, 0.38f, 48, 24), g_mesh_torus)) {
        std::cerr << "Failed to create Torus mesh\n"; return false;
    }
    if (!createGpuMesh(buildFloor(6.0f, 24, -1.3f), g_mesh_floor)) {
        std::cerr << "Failed to create Floor mesh\n"; return false;
    }
    if (!createGpuMesh(buildSphere(0.12f, 16, 16), g_mesh_sphere)) {
        std::cerr << "Failed to create Sphere mesh\n"; return false;
    }

    // 3. Динамический буфер линий для конусов прожекторов
    {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size  = MAX_LINE_VERTS * sizeof(Vertex);
        bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                  | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo ai;
        if (vmaCreateBuffer(ctx.allocator, &bci, &aci, &g_line_vb, &g_line_alloc, &ai) != VK_SUCCESS) {
            std::cerr << "Line buffer failed\n"; return false;
        }
        g_line_mapped = ai.pMappedData;
    }

    // 4. Буферы Сцены: UBO и Storage-буферы (SSBO) для источников света
    {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size  = sizeof(GpuSceneUBO);
        bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo ai;
        if (vmaCreateBuffer(ctx.allocator, &bci, &aci, &g_scene_ubo_buffer, &g_scene_ubo_alloc, &ai) != VK_SUCCESS) {
            std::cerr << "Scene UBO failed\n"; return false;
        }
        g_scene_ubo_mapped = ai.pMappedData;
    }

    {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size  = MAX_POINT_LIGHTS * sizeof(GpuPointLight);
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo ai;
        if (vmaCreateBuffer(ctx.allocator, &bci, &aci, &g_point_ssbo_buffer, &g_point_ssbo_alloc, &ai) != VK_SUCCESS) {
            std::cerr << "Point SSBO failed\n"; return false;
        }
        g_point_ssbo_mapped = ai.pMappedData;
    }

    {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size  = MAX_SPOT_LIGHTS * sizeof(GpuSpotLight);
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo ai;
        if (vmaCreateBuffer(ctx.allocator, &bci, &aci, &g_spot_ssbo_buffer, &g_spot_ssbo_alloc, &ai) != VK_SUCCESS) {
            std::cerr << "Spot SSBO failed\n"; return false;
        }
        g_spot_ssbo_mapped = ai.pMappedData;
    }

    // 5. Макеты наборов дескрипторов (Descriptor Set Layouts)
    // Set 0 (Сцена): Binding 0 - SceneUBO, Binding 1 - Point SSBO, Binding 2 - Spot SSBO
    {
        VkDescriptorSetLayoutBinding b[3]{};
        b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        b[0].descriptorCount = 1; b[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        b[1].binding = 1; b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[1].descriptorCount = 1; b[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        b[2].binding = 2; b[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[2].descriptorCount = 1; b[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = 3; ci.pBindings = b;
        if (vkCreateDescriptorSetLayout(ctx.device, &ci, nullptr, &g_scene_desc_layout) != VK_SUCCESS) {
            std::cerr << "Scene descriptor layout failed\n"; return false;
        }
    }

    // Set 1 (Модель): Binding 0 - ModelUBO
    {
        VkDescriptorSetLayoutBinding b{};
        b.binding = 0; b.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        b.descriptorCount = 1; b.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = 1; ci.pBindings = &b;
        if (vkCreateDescriptorSetLayout(ctx.device, &ci, nullptr, &g_model_desc_layout) != VK_SUCCESS) {
            std::cerr << "Model descriptor layout failed\n"; return false;
        }
    }

    // 6. Пул дескрипторов и создание дескрипторных сетов
    {
        VkDescriptorPoolSize ps[2]{};
        ps[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        ps[0].descriptorCount = 1 + MAX_MODEL_SLOTS;
        ps[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ps[1].descriptorCount = 2;

        VkDescriptorPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets = 1 + MAX_MODEL_SLOTS;
        pci.poolSizeCount = 2; pci.pPoolSizes = ps;
        if (vkCreateDescriptorPool(ctx.device, &pci, nullptr, &g_desc_pool) != VK_SUCCESS) {
            std::cerr << "Descriptor pool failed\n"; return false;
        }

        // Выделение Set 0 (Сцена)
        VkDescriptorSetAllocateInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = g_desc_pool; dai.descriptorSetCount = 1;
        dai.pSetLayouts = &g_scene_desc_layout;
        if (vkAllocateDescriptorSets(ctx.device, &dai, &g_scene_desc_set) != VK_SUCCESS) {
            std::cerr << "Scene descriptor set alloc failed\n"; return false;
        }

        VkDescriptorBufferInfo ubi{ g_scene_ubo_buffer, 0, sizeof(GpuSceneUBO) };
        VkDescriptorBufferInfo psi{ g_point_ssbo_buffer, 0, MAX_POINT_LIGHTS * sizeof(GpuPointLight) };
        VkDescriptorBufferInfo ssi{ g_spot_ssbo_buffer, 0, MAX_SPOT_LIGHTS * sizeof(GpuSpotLight) };

        VkWriteDescriptorSet wr[3]{};
        wr[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[0].dstSet = g_scene_desc_set; wr[0].dstBinding = 0; wr[0].descriptorCount = 1;
        wr[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; wr[0].pBufferInfo = &ubi;

        wr[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[1].dstSet = g_scene_desc_set; wr[1].dstBinding = 1; wr[1].descriptorCount = 1;
        wr[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr[1].pBufferInfo = &psi;

        wr[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[2].dstSet = g_scene_desc_set; wr[2].dstBinding = 2; wr[2].descriptorCount = 1;
        wr[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr[2].pBufferInfo = &ssi;

        vkUpdateDescriptorSets(ctx.device, 3, wr, 0, nullptr);

        // Выделение слотов для Set 1 (Модели)
        g_model_slots.resize(MAX_MODEL_SLOTS);
        for (uint32_t i = 0; i < MAX_MODEL_SLOTS; ++i) {
            auto& slot = g_model_slots[i];
            VkBufferCreateInfo bci{};
            bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bci.size  = sizeof(GpuModelUBO);
            bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
            bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO;
            aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            VmaAllocationInfo ai;
            if (vmaCreateBuffer(ctx.allocator, &bci, &aci, &slot.ubo, &slot.alloc, &ai) != VK_SUCCESS) {
                std::cerr << "Model UBO slot " << i << " failed\n"; return false;
            }
            slot.mapped = ai.pMappedData;

            VkDescriptorSetAllocateInfo mai{};
            mai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            mai.descriptorPool = g_desc_pool; mai.descriptorSetCount = 1;
            mai.pSetLayouts = &g_model_desc_layout;
            if (vkAllocateDescriptorSets(ctx.device, &mai, &slot.desc_set) != VK_SUCCESS) {
                std::cerr << "Model descriptor set alloc failed\n"; return false;
            }

            VkDescriptorBufferInfo mbi{ slot.ubo, 0, sizeof(GpuModelUBO) };
            VkWriteDescriptorSet mwr{};
            mwr.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            mwr.dstSet = slot.desc_set; mwr.dstBinding = 0; mwr.descriptorCount = 1;
            mwr.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; mwr.pBufferInfo = &mbi;
            vkUpdateDescriptorSets(ctx.device, 1, &mwr, 0, nullptr);
        }
    }

    // 7. Макет конвейера (Pipeline Layout): Set 0 = Scene, Set 1 = Model
    {
        VkDescriptorSetLayout set_layouts[2] = { g_scene_desc_layout, g_model_desc_layout };
        VkPipelineLayoutCreateInfo pli{};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 2; pli.pSetLayouts = set_layouts;
        if (vkCreatePipelineLayout(ctx.device, &pli, nullptr, &g_pipe_layout) != VK_SUCCESS) {
            std::cerr << "Pipeline layout failed\n"; return false;
        }
    }

    // 8. Графические конвейеры (Mesh Pipeline и Line Pipeline)
    {
        auto vc = readFile("shaders/torus.vert.spv");
        auto fc = readFile("shaders/torus.frag.spv");
        VkShaderModule vm = makeShader(vc), fm = makeShader(fc);

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;   stages[0].module = vm; stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = fm; stages[1].pName = "main";

        VkVertexInputBindingDescription bind{};
        bind.binding = 0; bind.stride = sizeof(Vertex); bind.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkVertexInputAttributeDescription attrs[3]{};
        attrs[0] = { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, (uint32_t)offsetof(Vertex, pos)    };
        attrs[1] = { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, (uint32_t)offsetof(Vertex, normal) };
        attrs[2] = { 2, 0, VK_FORMAT_R32G32B32_SFLOAT, (uint32_t)offsetof(Vertex, color)  };

        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount = 1;   vi.pVertexBindingDescriptions = &bind;
        vi.vertexAttributeDescriptionCount = 3; vi.pVertexAttributeDescriptions = attrs;

        VkPipelineViewportStateCreateInfo vps{};
        vps.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vps.viewportCount = 1; vps.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo raster{};
        raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.cullMode    = VK_CULL_MODE_NONE; // Двусторонний рендер
        raster.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        raster.lineWidth   = 1.f;

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp  = VK_COMPARE_OP_LESS;

        VkPipelineColorBlendAttachmentState ba{};
        ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo blend{};
        blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        blend.attachmentCount = 1; blend.pAttachments = &ba;

        VkDynamicState dyn_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dyn{};
        dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dyn.dynamicStateCount = 2; dyn.pDynamicStates = dyn_states;

        // Основной конвейер для полигональных мешей (треугольники)
        VkPipelineInputAssemblyStateCreateInfo ia_mesh{};
        ia_mesh.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia_mesh.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkGraphicsPipelineCreateInfo gci{};
        gci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gci.stageCount          = 2;           gci.pStages             = stages;
        gci.pVertexInputState   = &vi;         gci.pInputAssemblyState = &ia_mesh;
        gci.pViewportState      = &vps;        gci.pRasterizationState = &raster;
        gci.pMultisampleState   = &ms;         gci.pDepthStencilState  = &ds;
        gci.pColorBlendState    = &blend;      gci.pDynamicState       = &dyn;
        gci.layout              = g_pipe_layout;
        gci.renderPass          = ctx.render_pass;
        gci.subpass             = 0;

        if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &gci, nullptr, &g_mesh_pipeline) != VK_SUCCESS) {
            std::cerr << "Mesh pipeline creation failed\n"; return false;
        }

        // Конвейер для линий (отрисовка каркасных конусов прожекторов)
        VkPipelineInputAssemblyStateCreateInfo ia_line{};
        ia_line.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia_line.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        gci.pInputAssemblyState = &ia_line;

        if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &gci, nullptr, &g_line_pipeline) != VK_SUCCESS) {
            std::cerr << "Line pipeline creation failed\n"; return false;
        }

        vkDestroyShaderModule(ctx.device, vm, nullptr);
        vkDestroyShaderModule(ctx.device, fm, nullptr);
    }

    return true;
}

// ===========================================================================
// Освобождение ресурсов при закрытии
// ===========================================================================

void shutdown() {
    auto& ctx = graphics::internal::context;
    vkQueueWaitIdle(ctx.graphics_queue);

    if (g_line_pipeline ) vkDestroyPipeline(ctx.device, g_line_pipeline, nullptr);
    if (g_mesh_pipeline ) vkDestroyPipeline(ctx.device, g_mesh_pipeline, nullptr);
    if (g_pipe_layout   ) vkDestroyPipelineLayout(ctx.device, g_pipe_layout, nullptr);

    if (g_desc_pool) vkDestroyDescriptorPool(ctx.device, g_desc_pool, nullptr);
    if (g_model_desc_layout) vkDestroyDescriptorSetLayout(ctx.device, g_model_desc_layout, nullptr);
    if (g_scene_desc_layout) vkDestroyDescriptorSetLayout(ctx.device, g_scene_desc_layout, nullptr);

    for (auto& slot : g_model_slots) {
        if (slot.ubo) vmaDestroyBuffer(ctx.allocator, slot.ubo, slot.alloc);
    }
    g_model_slots.clear();

    if (g_spot_ssbo_buffer ) vmaDestroyBuffer(ctx.allocator, g_spot_ssbo_buffer,  g_spot_ssbo_alloc);
    if (g_point_ssbo_buffer) vmaDestroyBuffer(ctx.allocator, g_point_ssbo_buffer, g_point_ssbo_alloc);
    if (g_scene_ubo_buffer ) vmaDestroyBuffer(ctx.allocator, g_scene_ubo_buffer,  g_scene_ubo_alloc);
    if (g_line_vb          ) vmaDestroyBuffer(ctx.allocator, g_line_vb,           g_line_alloc);

    destroyGpuMesh(g_mesh_sphere);
    destroyGpuMesh(g_mesh_floor);
    destroyGpuMesh(g_mesh_torus);
}

// ===========================================================================
// Обновление состояния (ImGUI, расчет анимаций, обновление буферов)
// ===========================================================================

void update(double time) {
    if (g_prev_time < 0.0) g_prev_time = time;
    float dt = (float)(time - g_prev_time);
    g_prev_time = time;
    g_total_time += dt;

    // ---- Графический интерфейс ImGUI ----
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(440, 680), ImGuiCond_FirstUseEver);
    ImGui::Begin("Лабораторная №2: Камера и Освещение (Вариант 7)");

    // 1. Управление камерой
    if (ImGui::CollapsingHeader("Камера (Camera)", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::RadioButton("Орбитальная (Orbit)", &g_cam_mode, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Ручная (Manual)",     &g_cam_mode, 1);

        if (g_cam_mode == 0) {
            ImGui::SliderFloat("Дистанция##cam", &g_cam_dist, 1.5f, 15.0f);
            ImGui::SliderFloat("Азимут (Yaw)##cam", &g_cam_azimuth, -(float)M_PI, (float)M_PI);
            ImGui::SliderFloat("Угол (Pitch)##cam", &g_cam_elevation, -1.45f, 1.45f);
            ImGui::SliderFloat3("Цель (Target)##cam", &g_cam_target.x, -5.f, 5.f);
        } else {
            ImGui::SliderFloat3("Позиция##cam", &g_cam_pos.x, -10.f, 10.f);
            ImGui::SliderFloat3("Цель (Target)##cam", &g_cam_target.x, -5.f, 5.f);
        }

        ImGui::Separator();
        ImGui::RadioButton("Перспективная", &g_use_perspective, 1);
        ImGui::SameLine();
        ImGui::RadioButton("Ортографическая", &g_use_perspective, 0);

        if (g_use_perspective == 1) {
            ImGui::SliderFloat("Угол обзора (FOV)", &g_cam_fov_deg, 20.f, 110.f);
        } else {
            ImGui::SliderFloat("Размер орто##ortho", &g_ortho_size, 0.5f, 10.f);
        }

        if (ImGui::Button("Сбросить камеру")) {
            g_cam_dist = 5.5f;
            g_cam_azimuth = 0.45f;
            g_cam_elevation = 0.40f;
            g_cam_target = { 0.f, 0.f, 0.f };
            g_cam_pos = { 0.f, -5.f, 2.5f };
            g_cam_fov_deg = 45.f;
            g_use_perspective = 1;
        }
    }

    // 2. Направленный и фоновый свет
    if (ImGui::CollapsingHeader("Направленный и Фоновый свет", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TextColored(ImVec4(0.8f, 0.9f, 1.0f, 1.0f), "Фоновое освещение (Ambient Light):");
        ImGui::ColorEdit3("Цвет фона", g_ambient_color);
        ImGui::SliderFloat("Интенсивность фона", &g_ambient_intensity, 0.0f, 1.5f);

        ImGui::Separator();
        ImGui::TextColored(ImVec4(1.0f, 0.95f, 0.7f, 1.0f), "Направленный источник (Directional Light):");
        ImGui::Checkbox("Включить направленный##dir", &g_dir_enabled);
        ImGui::SliderFloat("Угол Pitch##dir", &g_dir_pitch, -1.5f, 1.5f);
        ImGui::SliderFloat("Угол Yaw##dir",   &g_dir_yaw,   -(float)M_PI, (float)M_PI);
        ImGui::ColorEdit3("Цвет света##dir", g_dir_color);
        ImGui::SliderFloat("Интенсивность##dir", &g_dir_intensity, 0.0f, 5.0f);
    }

    // 3. Точечные источники света (массив в SSBO)
    if (ImGui::CollapsingHeader("Точечные источники (Point Lights [SSBO])", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Всего точечных источников: %zu", g_point_lights.size());
        if (ImGui::Button("＋ Добавить точечный источник")) {
            PointLightItem npl;
            npl.position = { 0.0f, 1.8f, 1.0f };
            npl.color[0] = 0.3f; npl.color[1] = 1.0f; npl.color[2] = 0.4f; // Изумрудный
            npl.intensity = 3.0f;
            npl.animated = true;
            npl.animRadius = 2.0f;
            npl.animOffset = (float)g_point_lights.size() * 1.5f;
            g_point_lights.push_back(npl);
        }

        for (size_t i = 0; i < g_point_lights.size(); ++i) {
            auto& pl = g_point_lights[i];
            ImGui::PushID((int)i * 100);
            char headerTitle[64];
            snprintf(headerTitle, sizeof(headerTitle), "Точечный источник #%zu", i + 1);

            if (ImGui::TreeNode(headerTitle)) {
                ImGui::Checkbox("Включен", &pl.enabled);
                ImGui::SameLine();
                if (ImGui::Button("Удалить источник")) {
                    g_point_lights.erase(g_point_lights.begin() + i);
                    ImGui::TreePop();
                    ImGui::PopID();
                    break;
                }
                ImGui::SliderFloat3("Позиция", &pl.position.x, -6.f, 6.f);
                ImGui::ColorEdit3("Цвет", pl.color);
                ImGui::SliderFloat("Интенсивность", &pl.intensity, 0.0f, 10.0f);

                ImGui::TextDisabled("Затухание (Закон обратных квадратов):");
                ImGui::SliderFloat("Квадратичный коэфф. (q)", &pl.quadAtt, 0.001f, 0.5f, "%.4f");
                ImGui::SliderFloat("Линейный коэфф. (l)",      &pl.linAtt,  0.0f,   0.3f, "%.4f");
                ImGui::SliderFloat("Константный коэфф. (c)",   &pl.constAtt, 0.1f,  2.0f, "%.2f");

                ImGui::Checkbox("Орбитальная анимация", &pl.animated);
                if (pl.animated) {
                    ImGui::SliderFloat("Радиус орбиты", &pl.animRadius, 0.5f, 4.0f);
                    ImGui::SliderFloat("Скорость вращения", &pl.animSpeed, 0.1f, 4.0f);
                }
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
    }

    // 4. Прожекторы (массив в SSBO)
    if (ImGui::CollapsingHeader("Прожекторные источники (Spot Lights [SSBO])", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("Всего прожекторов: %zu", g_spot_lights.size());
        if (ImGui::Button("＋ Добавить прожектор")) {
            SpotLightItem nsl;
            nsl.position = { 0.0f, 2.0f, 2.0f };
            nsl.target   = { 0.0f, 0.0f, 0.0f };
            nsl.color[0] = 1.0f; nsl.color[1] = 0.9f; nsl.color[2] = 0.2f;
            nsl.intensity = 5.0f;
            nsl.innerAngleDeg = 15.0f;
            nsl.outerAngleDeg = 30.0f;
            g_spot_lights.push_back(nsl);
        }

        for (size_t i = 0; i < g_spot_lights.size(); ++i) {
            auto& sl = g_spot_lights[i];
            ImGui::PushID((int)i * 200 + 50);
            char headerTitle[64];
            snprintf(headerTitle, sizeof(headerTitle), "Прожектор #%zu", i + 1);

            if (ImGui::TreeNode(headerTitle)) {
                ImGui::Checkbox("Включен", &sl.enabled);
                ImGui::SameLine();
                if (ImGui::Button("Удалить прожектор")) {
                    g_spot_lights.erase(g_spot_lights.begin() + i);
                    ImGui::TreePop();
                    ImGui::PopID();
                    break;
                }
                ImGui::SliderFloat3("Позиция", &sl.position.x, -6.f, 6.f);
                ImGui::SliderFloat3("Точка цели (Target)", &sl.target.x, -5.f, 5.f);
                ImGui::ColorEdit3("Цвет", sl.color);
                ImGui::SliderFloat("Интенсивность", &sl.intensity, 0.0f, 15.0f);

                ImGui::TextDisabled("Мягкие края (Углы конуса):");
                ImGui::SliderFloat("Внутренний угол (Inner)", &sl.innerAngleDeg, 2.0f, 60.0f, "%.1f deg");
                ImGui::SliderFloat("Внешний угол (Outer)",     &sl.outerAngleDeg, 3.0f, 80.0f, "%.1f deg");
                if (sl.outerAngleDeg < sl.innerAngleDeg) {
                    sl.outerAngleDeg = sl.innerAngleDeg;
                }

                ImGui::SliderFloat("Затухание (квадратичное)", &sl.quadAtt, 0.001f, 0.2f, "%.4f");
                ImGui::Checkbox("Анимация покачивания", &sl.animated);
                if (sl.animated) {
                    ImGui::SliderFloat("Скорость анимации##sl", &sl.animSpeed, 0.1f, 3.0f);
                    ImGui::SliderFloat("Амплитуда покачивания", &sl.animSweep, 0.5f, 3.0f);
                }
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
    }

    // 5. Материалы Блинн-Фонга (Тор и Пол)
    if (ImGui::CollapsingHeader("Материалы (Блинн-Фонг)", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.4f, 1.0f), "Материал Тора:");
        ImGui::Checkbox("Радужная раскраска вершин", &g_torus_rainbow);
        ImGui::ColorEdit3("Альбедо (Albedo)##torus", g_torus_albedo);
        ImGui::ColorEdit3("Цвет блика (Specular)##torus", g_torus_specular);
        ImGui::SliderFloat("Коэфф. блика (Shininess)##torus", &g_torus_shininess, 2.0f, 256.0f, "%.1f");

        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.8f, 0.85f, 0.9f, 1.0f), "Материал Пола / Сцены:");
        ImGui::Checkbox("Отрисовывать пол сцены", &g_floor_enabled);
        if (g_floor_enabled) {
            ImGui::ColorEdit3("Альбедо пола##floor", g_floor_albedo);
            ImGui::ColorEdit3("Блик пола##floor", g_floor_specular);
            ImGui::SliderFloat("Коэфф. блика пола##floor", &g_floor_shininess, 2.0f, 128.0f, "%.1f");
        }
    }

    // 6. Трансформация меша тора
    if (ImGui::CollapsingHeader("Трансформация объекта")) {
        ImGui::SliderFloat3("Сдвиг (Translate)", g_model_pos, -4.f, 4.f);
        ImGui::SliderFloat3("Поворот (Euler)", g_model_rot, -(float)M_PI, (float)M_PI);
        ImGui::SliderFloat3("Масштаб (Scale)", g_model_scale, 0.1f, 3.f);
        ImGui::Checkbox("Автоматическое вращение тора", &g_model_spin);
    }

    // 7. Визуализация источников освещения (на оценку 5)
    if (ImGui::CollapsingHeader("Визуализация источников освещения", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Показывать сферы источников света", &g_show_gizmos);
        ImGui::Checkbox("Показывать конусы и направление прожекторов", &g_show_cones);
    }

    ImGui::Separator();
    ImGui::TextDisabled("Vulkan Blinn-Phong Renderer | 60 FPS");
    ImGui::End();

    // ---- Обновление параметров и анимаций ----
    // Орбитальная анимация точечных источников
    for (auto& pl : g_point_lights) {
        if (pl.animated) {
            float angle = g_total_time * pl.animSpeed + pl.animOffset;
            pl.position.x = pl.animRadius * std::cos(angle);
            pl.position.y = pl.animRadius * std::sin(angle);
            pl.position.z = 0.5f + 0.6f * std::sin(angle * 1.5f);
        }
    }

    // Покачивание прожекторов
    for (auto& sl : g_spot_lights) {
        if (sl.animated) {
            float sweep = std::sin(g_total_time * sl.animSpeed) * sl.animSweep;
            sl.target.x = sweep;
        }
    }

    // Вращение тора
    if (g_model_spin) {
        g_model_rot[2] += dt * 0.8f;
        if (g_model_rot[2] > (float)M_PI) g_model_rot[2] -= 2.f * (float)M_PI;
    }

    // Расчет положения камеры
    Vec3 eyePos;
    if (g_cam_mode == 0) {
        eyePos.x = g_cam_target.x + g_cam_dist * std::cos(g_cam_elevation) * std::sin(g_cam_azimuth);
        eyePos.y = g_cam_target.y - g_cam_dist * std::cos(g_cam_elevation) * std::cos(g_cam_azimuth);
        eyePos.z = g_cam_target.z + g_cam_dist * std::sin(g_cam_elevation);
    } else {
        eyePos = g_cam_pos;
    }

    auto& ctx = graphics::internal::context;
    float aspect = (float)ctx.swapchain_extent.width / (float)ctx.swapchain_extent.height;
    Mat4 view = mat4_look_at(eyePos, g_cam_target, { 0.f, 0.f, 1.f });
    Mat4 proj;
    if (g_use_perspective == 1) {
        proj = mat4_perspective(g_cam_fov_deg * (float)M_PI / 180.f, aspect, 0.1f, 100.f);
    } else {
        float hw = g_ortho_size * aspect;
        proj = mat4_ortho(-hw, hw, -g_ortho_size, g_ortho_size, 0.1f, 100.f);
    }
    Mat4 viewProj = mat4_mul(proj, view);

    // Расчет вектора направления направленного источника света
    float cp = std::cos(g_dir_pitch), sp = std::sin(g_dir_pitch);
    float cy = std::cos(g_dir_yaw),   sy = std::sin(g_dir_yaw);
    Vec3 dirVector = { cp * sy, cp * cy, -sp };

    // Заполнение структуры UBO Сцены
    GpuSceneUBO sceneUbo{};
    sceneUbo.view = view;
    sceneUbo.proj = proj;
    sceneUbo.viewProj = viewProj;
    sceneUbo.cameraPos[0] = eyePos.x;
    sceneUbo.cameraPos[1] = eyePos.y;
    sceneUbo.cameraPos[2] = eyePos.z;
    sceneUbo.cameraPos[3] = 1.0f;

    sceneUbo.ambientLight[0] = g_ambient_color[0];
    sceneUbo.ambientLight[1] = g_ambient_color[1];
    sceneUbo.ambientLight[2] = g_ambient_color[2];
    sceneUbo.ambientLight[3] = g_ambient_intensity;

    sceneUbo.dirLight.direction[0] = dirVector.x;
    sceneUbo.dirLight.direction[1] = dirVector.y;
    sceneUbo.dirLight.direction[2] = dirVector.z;
    sceneUbo.dirLight.direction[3] = g_dir_enabled ? 1.0f : 0.0f;

    sceneUbo.dirLight.color[0] = g_dir_color[0];
    sceneUbo.dirLight.color[1] = g_dir_color[1];
    sceneUbo.dirLight.color[2] = g_dir_color[2];
    sceneUbo.dirLight.color[3] = g_dir_intensity;

    sceneUbo.numPointLights = (int32_t)std::min<size_t>(g_point_lights.size(), MAX_POINT_LIGHTS);
    sceneUbo.numSpotLights  = (int32_t)std::min<size_t>(g_spot_lights.size(), MAX_SPOT_LIGHTS);
    memcpy(g_scene_ubo_mapped, &sceneUbo, sizeof(GpuSceneUBO));

    // Заполнение Storage-буфера точечных источников (SSBO)
    std::vector<GpuPointLight> gpuPoints(sceneUbo.numPointLights);
    for (int i = 0; i < sceneUbo.numPointLights; ++i) {
        const auto& src = g_point_lights[i];
        gpuPoints[i].position[0] = src.position.x;
        gpuPoints[i].position[1] = src.position.y;
        gpuPoints[i].position[2] = src.position.z;
        gpuPoints[i].position[3] = 1.0f;

        gpuPoints[i].color[0] = src.color[0];
        gpuPoints[i].color[1] = src.color[1];
        gpuPoints[i].color[2] = src.color[2];
        gpuPoints[i].color[3] = src.intensity;

        gpuPoints[i].params[0] = src.constAtt;
        gpuPoints[i].params[1] = src.linAtt;
        gpuPoints[i].params[2] = src.quadAtt;
        gpuPoints[i].params[3] = src.enabled ? 1.0f : 0.0f;
    }
    if (!gpuPoints.empty()) {
        memcpy(g_point_ssbo_mapped, gpuPoints.data(), gpuPoints.size() * sizeof(GpuPointLight));
    }

    // Заполнение Storage-буфера прожекторов (SSBO)
    std::vector<GpuSpotLight> gpuSpots(sceneUbo.numSpotLights);
    for (int i = 0; i < sceneUbo.numSpotLights; ++i) {
        const auto& src = g_spot_lights[i];
        gpuSpots[i].position[0] = src.position.x;
        gpuSpots[i].position[1] = src.position.y;
        gpuSpots[i].position[2] = src.position.z;
        gpuSpots[i].position[3] = 1.0f;

        Vec3 dir = vec3_normalize(src.target - src.position);
        gpuSpots[i].direction[0] = dir.x;
        gpuSpots[i].direction[1] = dir.y;
        gpuSpots[i].direction[2] = dir.z;
        gpuSpots[i].direction[3] = 0.0f;

        gpuSpots[i].color[0] = src.color[0];
        gpuSpots[i].color[1] = src.color[1];
        gpuSpots[i].color[2] = src.color[2];
        gpuSpots[i].color[3] = src.intensity;

        float innerCos = std::cos(src.innerAngleDeg * (float)M_PI / 180.f);
        float outerCos = std::cos(src.outerAngleDeg * (float)M_PI / 180.f);
        gpuSpots[i].params[0] = innerCos;
        gpuSpots[i].params[1] = outerCos;
        gpuSpots[i].params[2] = src.quadAtt;
        gpuSpots[i].params[3] = src.enabled ? 1.0f : 0.0f;
    }
    if (!gpuSpots.empty()) {
        memcpy(g_spot_ssbo_mapped, gpuSpots.data(), gpuSpots.size() * sizeof(GpuSpotLight));
    }
}

// ===========================================================================
// Отрисовка кадра (Render)
// ===========================================================================

void render(const graphics::internal::FrameData& fd) {
    auto& ctx = graphics::internal::context;
    VkCommandBuffer cmd = fd.command_buffer;

    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);

    VkClearValue clears[2]{};
    clears[0].color        = {{ 0.03f, 0.04f, 0.06f, 1.0f }};
    clears[1].depthStencil = { 1.f, 0 };

    VkRenderPassBeginInfo rp{};
    rp.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp.renderPass      = ctx.render_pass;
    rp.framebuffer     = fd.framebuffer;
    rp.renderArea      = { {0, 0}, ctx.swapchain_extent };
    rp.clearValueCount = 2;
    rp.pClearValues    = clears;

    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport vp{ 0.f, 0.f, (float)ctx.swapchain_extent.width, (float)ctx.swapchain_extent.height, 0.f, 1.f };
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D scissor{ {0, 0}, ctx.swapchain_extent };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // Привязываем общий дескрипторный набор сцены (Set 0)
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_pipe_layout, 0, 1, &g_scene_desc_set, 0, nullptr);

    // Привязываем пайплайн для полигональных мешей
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_mesh_pipeline);

    uint32_t currentSlot = 0;
    VkDeviceSize offset = 0;

    // 1. Отрисовка Тора (Вариант 7)
    {
        Mat4 model = mat4_mul(
            mat4_translate(g_model_pos[0], g_model_pos[1], g_model_pos[2]),
            mat4_mul(mat4_rotate_z(g_model_rot[2]),
            mat4_mul(mat4_rotate_y(g_model_rot[1]),
            mat4_mul(mat4_rotate_x(g_model_rot[0]),
                     mat4_scale(g_model_scale[0], g_model_scale[1], g_model_scale[2])))));

        GpuModelUBO torusUbo{};
        torusUbo.model = model;
        torusUbo.normalMatrix = mat4_normal_matrix(model);

        if (g_torus_rainbow) {
            torusUbo.albedo[0] = g_torus_albedo[0];
            torusUbo.albedo[1] = g_torus_albedo[1];
            torusUbo.albedo[2] = g_torus_albedo[2];
            torusUbo.albedo[3] = 1.0f;
        } else {
            torusUbo.albedo[0] = g_torus_albedo[0];
            torusUbo.albedo[1] = g_torus_albedo[1];
            torusUbo.albedo[2] = g_torus_albedo[2];
            torusUbo.albedo[3] = 0.0f; // Умножение на цвет вершин
        }

        torusUbo.specular[0] = g_torus_specular[0];
        torusUbo.specular[1] = g_torus_specular[1];
        torusUbo.specular[2] = g_torus_specular[2];
        torusUbo.specular[3] = g_torus_shininess;

        torusUbo.emissive[0] = 0.0f;
        torusUbo.emissive[1] = 0.0f;
        torusUbo.emissive[2] = 0.0f;
        torusUbo.emissive[3] = 0.0f; // Освещается по Блинн-Фонгу

        memcpy(g_model_slots[currentSlot].mapped, &torusUbo, sizeof(GpuModelUBO));
        // Привязываем дескрипторный набор модели (Set 1)
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_pipe_layout, 1, 1,
                                &g_model_slots[currentSlot].desc_set, 0, nullptr);
        currentSlot++;

        vkCmdBindVertexBuffers(cmd, 0, 1, &g_mesh_torus.vertexBuffer, &offset);
        vkCmdBindIndexBuffer(cmd, g_mesh_torus.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, g_mesh_torus.indexCount, 1, 0, 0, 0);
    }

    // 2. Отрисовка пола сцены
    if (g_floor_enabled && currentSlot < MAX_MODEL_SLOTS) {
        Mat4 model = Mat4::identity();
        GpuModelUBO floorUbo{};
        floorUbo.model = model;
        floorUbo.normalMatrix = Mat4::identity();
        floorUbo.albedo[0] = g_floor_albedo[0];
        floorUbo.albedo[1] = g_floor_albedo[1];
        floorUbo.albedo[2] = g_floor_albedo[2];
        floorUbo.albedo[3] = 1.0f;

        floorUbo.specular[0] = g_floor_specular[0];
        floorUbo.specular[1] = g_floor_specular[1];
        floorUbo.specular[2] = g_floor_specular[2];
        floorUbo.specular[3] = g_floor_shininess;

        floorUbo.emissive[0] = 0.0f;
        floorUbo.emissive[1] = 0.0f;
        floorUbo.emissive[2] = 0.0f;
        floorUbo.emissive[3] = 0.0f; // Освещается по Блинн-Фонгу

        memcpy(g_model_slots[currentSlot].mapped, &floorUbo, sizeof(GpuModelUBO));
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_pipe_layout, 1, 1,
                                &g_model_slots[currentSlot].desc_set, 0, nullptr);
        currentSlot++;

        vkCmdBindVertexBuffers(cmd, 0, 1, &g_mesh_floor.vertexBuffer, &offset);
        vkCmdBindIndexBuffer(cmd, g_mesh_floor.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, g_mesh_floor.indexCount, 1, 0, 0, 0);
    }

    // 3. Отрисовка маркеров источников света (сферы с чистым unlit цветом)
    if (g_show_gizmos) {
        vkCmdBindVertexBuffers(cmd, 0, 1, &g_mesh_sphere.vertexBuffer, &offset);
        vkCmdBindIndexBuffer(cmd, g_mesh_sphere.indexBuffer, 0, VK_INDEX_TYPE_UINT32);

        // Маркеры точечных источников
        for (const auto& pl : g_point_lights) {
            if (!pl.enabled || currentSlot >= MAX_MODEL_SLOTS) continue;
            Mat4 model = mat4_translate(pl.position.x, pl.position.y, pl.position.z);
            GpuModelUBO gizmoUbo{};
            gizmoUbo.model = model;
            gizmoUbo.normalMatrix = Mat4::identity();
            gizmoUbo.emissive[0] = pl.color[0];
            gizmoUbo.emissive[1] = pl.color[1];
            gizmoUbo.emissive[2] = pl.color[2];
            gizmoUbo.emissive[3] = 1.0f; // Unlit (чистый цвет без Блинн-Фонга)

            memcpy(g_model_slots[currentSlot].mapped, &gizmoUbo, sizeof(GpuModelUBO));
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_pipe_layout, 1, 1,
                                    &g_model_slots[currentSlot].desc_set, 0, nullptr);
            currentSlot++;
            vkCmdDrawIndexed(cmd, g_mesh_sphere.indexCount, 1, 0, 0, 0);
        }

        // Маркеры прожекторов
        for (const auto& sl : g_spot_lights) {
            if (!sl.enabled || currentSlot >= MAX_MODEL_SLOTS) continue;
            Mat4 model = mat4_translate(sl.position.x, sl.position.y, sl.position.z);
            GpuModelUBO gizmoUbo{};
            gizmoUbo.model = model;
            gizmoUbo.normalMatrix = Mat4::identity();
            gizmoUbo.emissive[0] = sl.color[0];
            gizmoUbo.emissive[1] = sl.color[1];
            gizmoUbo.emissive[2] = sl.color[2];
            gizmoUbo.emissive[3] = 1.0f; // Unlit

            memcpy(g_model_slots[currentSlot].mapped, &gizmoUbo, sizeof(GpuModelUBO));
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_pipe_layout, 1, 1,
                                    &g_model_slots[currentSlot].desc_set, 0, nullptr);
            currentSlot++;
            vkCmdDrawIndexed(cmd, g_mesh_sphere.indexCount, 1, 0, 0, 0);
        }
    }

    // 4. Отрисовка белых каркасных конусов и луча направления прожекторов
    if (g_show_cones && currentSlot < MAX_MODEL_SLOTS) {
        std::vector<Vertex> lineVerts;
        const int SEGMENTS = 24;
        const float CONE_LEN = 2.4f;

        for (const auto& sl : g_spot_lights) {
            if (!sl.enabled) continue;
            Vec3 P = sl.position;
            Vec3 D = vec3_normalize(sl.target - sl.position);
            Vec3 C = P + D * CONE_LEN;

            float outerRad = sl.outerAngleDeg * (float)M_PI / 180.f;
            float R = CONE_LEN * std::tan(outerRad);

            Vec3 up = (std::abs(D.z) < 0.95f) ? Vec3{ 0.f, 0.f, 1.f } : Vec3{ 0.f, 1.f, 0.f };
            Vec3 U = vec3_normalize(vec3_cross(D, up));
            Vec3 V = vec3_cross(D, U);

            // Центральная линия направления
            Vertex v0, v1;
            v0.pos[0] = P.x; v0.pos[1] = P.y; v0.pos[2] = P.z;
            v0.color[0] = 1.0f; v0.color[1] = 0.9f; v0.color[2] = 0.3f;
            v1.pos[0] = C.x; v1.pos[1] = C.y; v1.pos[2] = C.z;
            v1.color[0] = 1.0f; v1.color[1] = 0.9f; v1.color[2] = 0.3f;
            lineVerts.push_back(v0); lineVerts.push_back(v1);

            // Окружность основания конуса и образующие лучи
            Vec3 prevQ;
            for (int k = 0; k <= SEGMENTS; ++k) {
                float a = 2.f * (float)M_PI * k / SEGMENTS;
                Vec3 Q = C + (U * std::cos(a) + V * std::sin(a)) * R;

                if (k > 0) {
                    Vertex va, vb;
                    va.pos[0] = prevQ.x; va.pos[1] = prevQ.y; va.pos[2] = prevQ.z;
                    va.color[0] = 1.f;   va.color[1] = 1.f;   va.color[2] = 1.f;
                    vb.pos[0] = Q.x;     vb.pos[1] = Q.y;     vb.pos[2] = Q.z;
                    vb.color[0] = 1.f;   vb.color[1] = 1.f;   vb.color[2] = 1.f;
                    lineVerts.push_back(va); lineVerts.push_back(vb);
                }
                prevQ = Q;

                // Образующие от вершины к основанию (каждые 6 сегментов)
                if (k < SEGMENTS && (k % 6 == 0)) {
                    Vertex va, vb;
                    va.pos[0] = P.x; va.pos[1] = P.y; va.pos[2] = P.z;
                    va.color[0] = 0.8f; va.color[1] = 0.8f; va.color[2] = 0.8f;
                    vb.pos[0] = Q.x; vb.pos[1] = Q.y; vb.pos[2] = Q.z;
                    vb.color[0] = 0.8f; vb.color[1] = 0.8f; vb.color[2] = 0.8f;
                    lineVerts.push_back(va); lineVerts.push_back(vb);
                }
            }
        }

        if (!lineVerts.empty()) {
            uint32_t count = (uint32_t)std::min<size_t>(lineVerts.size(), MAX_LINE_VERTS);
            memcpy(g_line_mapped, lineVerts.data(), count * sizeof(Vertex));

            GpuModelUBO lineUbo{};
            lineUbo.model = Mat4::identity();
            lineUbo.normalMatrix = Mat4::identity();
            lineUbo.emissive[0] = 1.0f;
            lineUbo.emissive[1] = 1.0f;
            lineUbo.emissive[2] = 1.0f;
            lineUbo.emissive[3] = 1.0f; // Unlit белые линии

            memcpy(g_model_slots[currentSlot].mapped, &lineUbo, sizeof(GpuModelUBO));
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_line_pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_pipe_layout, 1, 1,
                                    &g_model_slots[currentSlot].desc_set, 0, nullptr);
            currentSlot++;

            vkCmdBindVertexBuffers(cmd, 0, 1, &g_line_vb, &offset);
            vkCmdDraw(cmd, count, 1, 0, 0);
        }
    }

    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
}

} // namespace application