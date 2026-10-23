#include "application.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <string>

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <imgui.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace application {

// ===========================================================================
// Math  (column-major: m[col][row], matching GLSL/Vulkan convention)
// ===========================================================================

struct Vec3 { float x, y, z; };

struct Mat4 {
    float m[4][4]{};
    static Mat4 identity() {
        Mat4 r;
        r.m[0][0] = r.m[1][1] = r.m[2][2] = r.m[3][3] = 1.f;
        return r;
    }
};

// Column-major matrix multiply: result = a * b
static Mat4 mat4_mul(const Mat4& a, const Mat4& b) {
    Mat4 r;
    for (int col = 0; col < 4; ++col)
        for (int row = 0; row < 4; ++row)
            for (int k = 0; k < 4; ++k)
                r.m[col][row] += a.m[k][row] * b.m[col][k];
    return r;
}

static Mat4 mat4_translate(float tx, float ty, float tz) {
    Mat4 r = Mat4::identity();
    r.m[3][0] = tx; r.m[3][1] = ty; r.m[3][2] = tz;
    return r;
}

static Mat4 mat4_scale(float sx, float sy, float sz) {
    Mat4 r = Mat4::identity();
    r.m[0][0] = sx; r.m[1][1] = sy; r.m[2][2] = sz;
    return r;
}

static Mat4 mat4_rotate_x(float a) {
    Mat4 r = Mat4::identity();
    r.m[1][1] =  std::cos(a); r.m[1][2] =  std::sin(a);
    r.m[2][1] = -std::sin(a); r.m[2][2] =  std::cos(a);
    return r;
}

static Mat4 mat4_rotate_y(float a) {
    Mat4 r = Mat4::identity();
    r.m[0][0] =  std::cos(a); r.m[0][2] = -std::sin(a);
    r.m[2][0] =  std::sin(a); r.m[2][2] =  std::cos(a);
    return r;
}

static Mat4 mat4_rotate_z(float a) {
    Mat4 r = Mat4::identity();
    r.m[0][0] =  std::cos(a); r.m[0][1] =  std::sin(a);
    r.m[1][0] = -std::sin(a); r.m[1][1] =  std::cos(a);
    return r;
}

// Perspective projection (Vulkan depth [0,1], Y axis flipped)
static Mat4 mat4_perspective(float fovY_rad, float aspect, float near_z, float far_z) {
    float f = 1.f / std::tan(fovY_rad * 0.5f);
    Mat4 r;
    r.m[0][0] = f / aspect;
    r.m[1][1] = -f;                                         // Vulkan Y flip
    r.m[2][2] = far_z / (near_z - far_z);
    r.m[3][2] = (near_z * far_z) / (near_z - far_z);
    r.m[2][3] = -1.f;
    return r;
}

// Orthographic projection (Vulkan depth [0,1], Y axis flipped)
static Mat4 mat4_ortho(float left, float right, float bottom, float top,
                       float near_z, float far_z) {
    Mat4 r;
    r.m[0][0] =  2.f / (right - left);
    r.m[1][1] = -2.f / (top - bottom);   // Vulkan Y flip
    r.m[2][2] =  1.f / (near_z - far_z);
    r.m[3][0] = -(right + left)   / (right - left);
    r.m[3][1] =  (top   + bottom) / (top   - bottom);
    r.m[3][2] =  near_z / (near_z - far_z);
    r.m[3][3] =  1.f;
    return r;
}

static Mat4 mat4_look_at(Vec3 eye, Vec3 center, Vec3 up) {
    Vec3 f = { center.x - eye.x, center.y - eye.y, center.z - eye.z };
    float fl = std::sqrt(f.x*f.x + f.y*f.y + f.z*f.z);
    f = { f.x/fl, f.y/fl, f.z/fl };

    Vec3 rr = { f.y*up.z - f.z*up.y, f.z*up.x - f.x*up.z, f.x*up.y - f.y*up.x };
    float rl = std::sqrt(rr.x*rr.x + rr.y*rr.y + rr.z*rr.z);
    rr = { rr.x/rl, rr.y/rl, rr.z/rl };

    Vec3 u = { rr.y*f.z - rr.z*f.y, rr.z*f.x - rr.x*f.z, rr.x*f.y - rr.y*f.x };

    Mat4 mat = Mat4::identity();
    mat.m[0][0]=rr.x; mat.m[1][0]=rr.y; mat.m[2][0]=rr.z; mat.m[3][0]=-(rr.x*eye.x+rr.y*eye.y+rr.z*eye.z);
    mat.m[0][1]=u.x;  mat.m[1][1]=u.y;  mat.m[2][1]=u.z;  mat.m[3][1]=-(u.x*eye.x+u.y*eye.y+u.z*eye.z);
    mat.m[0][2]=-f.x; mat.m[1][2]=-f.y; mat.m[2][2]=-f.z; mat.m[3][2]= (f.x*eye.x+f.y*eye.y+f.z*eye.z);
    mat.m[3][3]=1.f;
    return mat;
}

// ===========================================================================
// Геометрия (Geometry)
// ===========================================================================

// Вершина (Vertex) - базовая единица 3D-модели.
// Хранит пакет данных о конкретной точке в пространстве.
struct Vertex {
    float pos[3];     // Позиция (X, Y, Z) в локальных координатах
    float normal[3];  // Нормаль (вектор направления) - нужна для расчёта освещения
    float color[3];   // Цвет (R, G, B) - Задание 5: процедурный цвет для каждой вершины
};

static std::vector<Vertex>   g_vertices;
static std::vector<uint32_t> g_indices;

// Simple HSV->RGB (S=1, V=1), h in [0,1]
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

static void buildTorus(float R, float r, int segments, int sides) {
    g_vertices.clear();
    g_indices.clear();

    for (int i = 0; i <= segments; ++i) {
        float theta = 2.f * (float)M_PI * i / segments;
        float ct = std::cos(theta), st = std::sin(theta);

        // Task 5: rainbow color along major ring (theta angle)
        float cr, cg, cb;
        hsv_to_rgb(theta / (2.f * (float)M_PI), cr, cg, cb);

        for (int j = 0; j <= sides; ++j) {
            float phi = 2.f * (float)M_PI * j / sides;
            float cp = std::cos(phi), sp = std::sin(phi);

            Vertex v;
            v.pos[0]    = (R + r * cp) * ct;
            v.pos[1]    = (R + r * cp) * st;
            v.pos[2]    = r * sp;
            v.normal[0] = cp * ct;
            v.normal[1] = cp * st;
            v.normal[2] = sp;
            v.color[0]  = cr;
            v.color[1]  = cg;
            v.color[2]  = cb;
            g_vertices.push_back(v);
        }
    }

    for (int i = 0; i < segments; ++i) {
        for (int j = 0; j < sides; ++j) {
            int a = i * (sides + 1) + j;
            int b = a + 1;
            int c = (i + 1) * (sides + 1) + j;
            int d = c + 1;
            g_indices.push_back(a); g_indices.push_back(c); g_indices.push_back(b);
            g_indices.push_back(b); g_indices.push_back(c); g_indices.push_back(d);
        }
    }
}

// ===========================================================================
// Ресурсы Vulkan (Vulkan resources)
// ===========================================================================

// UBO (Uniform Buffer Object) - структура данных, которая передается 
// из памяти CPU в память видеокарты (GPU) каждый кадр.
// Содержит матрицы трансформации (MVP) и пользовательские параметры (цвет).
struct UBO {
    Mat4  model;         // Матрица Модели (положение объекта в мире)
    Mat4  view;          // Матрица Вида (камера)
    Mat4  proj;          // Матрица Проекции (перевод в 2D экран)
    float user_color[4]; // Цвет, выбранный в ImGui (RGBA). Выровнен по 16 байт по правилам Vulkan
};
static_assert(sizeof(UBO) == 3 * 64 + 16, "UBO size mismatch");

static VkBuffer          g_vertex_buffer  = VK_NULL_HANDLE;
static VmaAllocation     g_vertex_alloc   = VK_NULL_HANDLE;
static VkBuffer          g_index_buffer   = VK_NULL_HANDLE;
static VmaAllocation     g_index_alloc    = VK_NULL_HANDLE;
static VkBuffer          g_ubo_buffer     = VK_NULL_HANDLE;
static VmaAllocation     g_ubo_alloc      = VK_NULL_HANDLE;
static void*             g_ubo_mapped     = nullptr;

static VkDescriptorSetLayout g_desc_layout = VK_NULL_HANDLE;
static VkDescriptorPool      g_desc_pool   = VK_NULL_HANDLE;
static VkDescriptorSet       g_desc_set    = VK_NULL_HANDLE;
static VkPipelineLayout      g_pipe_layout = VK_NULL_HANDLE;
static VkPipeline            g_pipeline    = VK_NULL_HANDLE;

// ===========================================================================
// Состояние приложения (Глобальные переменные для интерфейса и логики)
// ===========================================================================

// Задание 1: Переключение матрицы проекции
static int   g_use_perspective = 1; // 1 = Перспективная проекция, 0 = Ортографическая (int для кнопок ImGui)
static float g_ortho_size      = 2.5f; // Размер видимой области для ортографической проекции

// Задание 2: Элементы интерфейса для сдвига, поворота и масштаба
static float g_pos[3]   = { 0.f, 0.f, 0.f }; // Позиция (Translate)
static float g_rot[3]   = { 0.f, 0.f, 0.f }; // Углы поворота (Rotate) в радианах по X/Y/Z
static float g_scale[3] = { 1.f, 1.f, 1.f }; // Масштаб (Scale)

// Задание 3: Анимация и движение по сложной траектории
static bool  g_anim_playing = false;  // Флаг воспроизведения (Play/Pause)
static float g_anim_speed   = 1.f;    // Скорость анимации
static float g_anim_radius  = 0.f;    // Радиус вращения (орбиты)
static float g_anim_height  = 0.f;    // Высота подпрыгивания (амплитуда по оси Z)
static float g_anim_spin    = 1.f;    // Вращение объекта вокруг своей оси Y во время полета
static float g_anim_time    = 0.f;    // Текущее время анимации (накапливается каждый кадр)
static double g_prev_time   = -1.0;   // Время предыдущего кадра для расчета delta time (dt)

// Задание 4: Возможность изменения цвета фигуры из интерфейса
static float g_user_color[4] = { 1.f, 1.f, 1.f, 1.f }; // Базовый цвет объекта (RGBA)

// ===========================================================================
// Helpers
// ===========================================================================

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

// ===========================================================================
// initialize
// ===========================================================================

bool initialize() {
    auto& ctx = graphics::internal::context;

    buildTorus(1.0f, 0.4f, 48, 24);

    // Vertex buffer
    VkDeviceSize vb_sz = g_vertices.size() * sizeof(Vertex);
    if (!makeDevBuf(vb_sz,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            g_vertex_buffer, g_vertex_alloc) ||
        !upload(g_vertex_buffer, g_vertices.data(), vb_sz)) {
        std::cerr << "Vertex buffer failed\n"; return false;
    }

    // Index buffer
    VkDeviceSize ib_sz = g_indices.size() * sizeof(uint32_t);
    if (!makeDevBuf(ib_sz,
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            g_index_buffer, g_index_alloc) ||
        !upload(g_index_buffer, g_indices.data(), ib_sz)) {
        std::cerr << "Index buffer failed\n"; return false;
    }

    // UBO (persistently mapped, host-visible)
    {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size  = sizeof(UBO);
        bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                  | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo ai;
        if (vmaCreateBuffer(ctx.allocator, &bci, &aci, &g_ubo_buffer, &g_ubo_alloc, &ai) != VK_SUCCESS) {
            std::cerr << "UBO failed\n"; return false;
        }
        g_ubo_mapped = ai.pMappedData;
    }

    // Descriptor set layout
    {
        VkDescriptorSetLayoutBinding b{};
        b.binding = 0; b.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        b.descriptorCount = 1; b.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = 1; ci.pBindings = &b;
        if (vkCreateDescriptorSetLayout(ctx.device, &ci, nullptr, &g_desc_layout) != VK_SUCCESS) {
            std::cerr << "Descriptor set layout failed\n"; return false;
        }
    }

    // Descriptor pool + set
    {
        VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 };
        VkDescriptorPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets = 1; pci.poolSizeCount = 1; pci.pPoolSizes = &ps;
        if (vkCreateDescriptorPool(ctx.device, &pci, nullptr, &g_desc_pool) != VK_SUCCESS) {
            std::cerr << "Descriptor pool failed\n"; return false;
        }
        VkDescriptorSetAllocateInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = g_desc_pool; dai.descriptorSetCount = 1;
        dai.pSetLayouts = &g_desc_layout;
        if (vkAllocateDescriptorSets(ctx.device, &dai, &g_desc_set) != VK_SUCCESS) {
            std::cerr << "Descriptor set alloc failed\n"; return false;
        }
        VkDescriptorBufferInfo bi{ g_ubo_buffer, 0, sizeof(UBO) };
        VkWriteDescriptorSet wr{};
        wr.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr.dstSet = g_desc_set; wr.dstBinding = 0; wr.descriptorCount = 1;
        wr.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; wr.pBufferInfo = &bi;
        vkUpdateDescriptorSets(ctx.device, 1, &wr, 0, nullptr);
    }

    // Pipeline layout
    {
        VkPipelineLayoutCreateInfo pli{};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 1; pli.pSetLayouts = &g_desc_layout;
        if (vkCreatePipelineLayout(ctx.device, &pli, nullptr, &g_pipe_layout) != VK_SUCCESS) {
            std::cerr << "Pipeline layout failed\n"; return false;
        }
    }

    // Graphics pipeline
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

        // 3 vertex attributes: position, normal, color
        VkVertexInputAttributeDescription attrs[3]{};
        attrs[0] = { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, (uint32_t)offsetof(Vertex, pos)    };
        attrs[1] = { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, (uint32_t)offsetof(Vertex, normal) };
        attrs[2] = { 2, 0, VK_FORMAT_R32G32B32_SFLOAT, (uint32_t)offsetof(Vertex, color)  };

        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount = 1;   vi.pVertexBindingDescriptions = &bind;
        vi.vertexAttributeDescriptionCount = 3; vi.pVertexAttributeDescriptions = attrs;

        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo vps{};
        vps.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vps.viewportCount = 1; vps.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo raster{};
        raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.cullMode    = VK_CULL_MODE_BACK_BIT;
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

        VkGraphicsPipelineCreateInfo gci{};
        gci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gci.stageCount          = 2;           gci.pStages             = stages;
        gci.pVertexInputState   = &vi;         gci.pInputAssemblyState = &ia;
        gci.pViewportState      = &vps;        gci.pRasterizationState = &raster;
        gci.pMultisampleState   = &ms;         gci.pDepthStencilState  = &ds;
        gci.pColorBlendState    = &blend;      gci.pDynamicState       = &dyn;
        gci.layout              = g_pipe_layout;
        gci.renderPass          = ctx.render_pass;
        gci.subpass             = 0;

        VkResult res = vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &gci, nullptr, &g_pipeline);
        vkDestroyShaderModule(ctx.device, vm, nullptr);
        vkDestroyShaderModule(ctx.device, fm, nullptr);
        if (res != VK_SUCCESS) { std::cerr << "Pipeline failed\n"; return false; }
    }

    return true;
}

// ===========================================================================
// shutdown
// ===========================================================================

void shutdown() {
    auto& ctx = graphics::internal::context;
    vkQueueWaitIdle(ctx.graphics_queue);
    if (g_pipeline    ) vkDestroyPipeline          (ctx.device, g_pipeline,    nullptr);
    if (g_pipe_layout ) vkDestroyPipelineLayout     (ctx.device, g_pipe_layout, nullptr);
    if (g_desc_pool   ) vkDestroyDescriptorPool     (ctx.device, g_desc_pool,   nullptr);
    if (g_desc_layout ) vkDestroyDescriptorSetLayout(ctx.device, g_desc_layout, nullptr);
    if (g_ubo_buffer  ) vmaDestroyBuffer(ctx.allocator, g_ubo_buffer,    g_ubo_alloc);
    if (g_index_buffer) vmaDestroyBuffer(ctx.allocator, g_index_buffer,  g_index_alloc);
    if (g_vertex_buffer)vmaDestroyBuffer(ctx.allocator, g_vertex_buffer, g_vertex_alloc);
}

// ===========================================================================
// update — ImGUI + UBO
// ===========================================================================

void update(double time) {
    // Delta time
    if (g_prev_time < 0.0) g_prev_time = time;
    float dt = (float)(time - g_prev_time);
    g_prev_time = time;

    // Task 3: advance animation clock
    if (g_anim_playing) g_anim_time += dt * g_anim_speed;

    // ---- ImGUI ----
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(320, 470), ImGuiCond_FirstUseEver);
    ImGui::Begin("Тор (вариант 7)");

    // Task 1: projection switch
    if (ImGui::CollapsingHeader("Проекция", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::RadioButton("Перспективная",    &g_use_perspective, 1);
        ImGui::SameLine();
        ImGui::RadioButton("Ортографическая", &g_use_perspective, 0);
        if (g_use_perspective == 0)
            ImGui::SliderFloat("Размер##ortho", &g_ortho_size, 0.5f, 6.f);
    }

    // Task 2: position / rotation / scale
    if (ImGui::CollapsingHeader("Трансформация", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat3("Позиция",  g_pos,   -5.f, 5.f);
        ImGui::SliderFloat("Поворот X", &g_rot[0], -(float)M_PI, (float)M_PI);
        ImGui::SliderFloat("Поворот Y", &g_rot[1], -(float)M_PI, (float)M_PI);
        ImGui::SliderFloat("Поворот Z", &g_rot[2], -(float)M_PI, (float)M_PI);
        ImGui::SliderFloat3("Масштаб",  g_scale, 0.05f, 5.f);
    }

    // Task 3: animation
    if (ImGui::CollapsingHeader("Анимация")) {
        if (g_anim_playing) {
            if (ImGui::Button("  Пауза  ##anim")) g_anim_playing = false;
        } else {
            if (ImGui::Button("Воспроизведение##anim")) g_anim_playing = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Сброс##anim")) g_anim_time = 0.f;
        ImGui::SliderFloat("Скорость##anim",       &g_anim_speed,  0.1f, 5.f);
        ImGui::SliderFloat("Радиус орбиты##anim",  &g_anim_radius, 0.f,  3.f);
        ImGui::SliderFloat("Амплитуда Z##anim",    &g_anim_height, 0.f,  2.f);
        ImGui::SliderFloat("Спин Y (об/оборот)##anim", &g_anim_spin, 0.f, 4.f);
    }

    // Task 4: color picker
    if (ImGui::CollapsingHeader("Цвет", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::ColorEdit4("Цвет фигуры", g_user_color);
    }

    ImGui::Separator();
    ImGui::TextDisabled("Вершин: %zu  Индексов: %zu", g_vertices.size(), g_indices.size());
    ImGui::End();

    // ---- Построение Матрицы Модели (Model Matrix) ----
    // Траектория движения (Задание 3): Фигуры Лиссажу (круг/эллипс + колебания по Z)
    float anim_x  = g_anim_radius * std::cos(g_anim_time);
    float anim_y  = g_anim_radius * std::sin(g_anim_time);
    float anim_z  = g_anim_height * std::sin(2.f * g_anim_time);
    float anim_ry = g_anim_spin   * g_anim_time; // Автоматическое вращение вокруг своей оси Y

    // Порядок умножения матриц TRS (Translate * Rotate * Scale) важен!
    // Читается справа налево: сначала масштабируем, затем вращаем (по X, Y, Z), затем сдвигаем.
    Mat4 model = mat4_mul(
        mat4_translate(g_pos[0] + anim_x, g_pos[1] + anim_y, g_pos[2] + anim_z),
        mat4_mul(mat4_rotate_z(g_rot[2]),
        mat4_mul(mat4_rotate_y(g_rot[1] + anim_ry),
        mat4_mul(mat4_rotate_x(g_rot[0]),
                 mat4_scale(g_scale[0], g_scale[1], g_scale[2])))));

    // ---- Построение Матрицы Проекции (Projection Matrix) - Задание 1 ----
    auto& ctx = graphics::internal::context;
    float aspect = (float)ctx.swapchain_extent.width / (float)ctx.swapchain_extent.height; // Соотношение сторон экрана
    Mat4 proj;
    if (g_use_perspective == 1) {
        // Перспективная проекция: объекты вдали кажутся меньше (FOV = 45 градусов, ближняя и дальняя плоскости отсечения)
        proj = mat4_perspective((float)M_PI / 4.f, aspect, 0.1f, 100.f);
    } else {
        // Ортографическая проекция: нет эффекта глубины (отличный выбор для чертежей, изометрии)
        float hw = g_ortho_size * aspect;
        proj = mat4_ortho(-hw, hw, -g_ortho_size, g_ortho_size, 0.1f, 100.f);
    }

    UBO ubo{};
    ubo.model = model;
    ubo.view  = mat4_look_at({0.f, -4.f, 2.f}, {0.f, 0.f, 0.f}, {0.f, 0.f, 1.f});
    ubo.proj  = proj;
    ubo.user_color[0] = g_user_color[0];
    ubo.user_color[1] = g_user_color[1];
    ubo.user_color[2] = g_user_color[2];
    ubo.user_color[3] = g_user_color[3];
    memcpy(g_ubo_mapped, &ubo, sizeof(ubo));
}

// ===========================================================================
// render
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
    clears[0].color        = {{ 0.04f, 0.04f, 0.07f, 1.0f }};
    clears[1].depthStencil = { 1.f, 0 };

    VkRenderPassBeginInfo rp{};
    rp.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp.renderPass      = ctx.render_pass;
    rp.framebuffer     = fd.framebuffer;
    rp.renderArea      = { {0, 0}, ctx.swapchain_extent };
    rp.clearValueCount = 2;
    rp.pClearValues    = clears;

    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_pipeline);

    VkViewport vp{ 0.f, 0.f,
        (float)ctx.swapchain_extent.width, (float)ctx.swapchain_extent.height,
        0.f, 1.f };
    vkCmdSetViewport(cmd, 0, 1, &vp);

    VkRect2D scissor{ {0, 0}, ctx.swapchain_extent };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &g_vertex_buffer, &offset);
    vkCmdBindIndexBuffer(cmd, g_index_buffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            g_pipe_layout, 0, 1, &g_desc_set, 0, nullptr);
    vkCmdDrawIndexed(cmd, (uint32_t)g_indices.size(), 1, 0, 0, 0);

    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
}

} // namespace application