#ifndef _BYTE_BUFFER_HPP_
#define _BYTE_BUFFER_HPP_

#include <stdint.h>
#include <cstring>
#include <cassert>
#include <string_view>
#include <algorithm>
#include <span>
#include <utility>

class ByteBuffer {
public:
    // 0. 默认构造（空缓冲区）
    ByteBuffer() noexcept : data_(nullptr), size_(0), owns_(false) {}

    // 1. 持有模式：分配内存
    ByteBuffer(size_t size) 
        : data_(new uint8_t[size]()), size_(size), owns_(true) {}

    // 2. 非持有模式：借用外部内存（指针必须有效且生命周期长于本对象）
    ByteBuffer(uint8_t* data, size_t size) 
        : data_(data), size_(size), owns_(false) {}

    // 3. 从 string_view 借用（常见场景）
    ByteBuffer(std::string_view sv) 
        : data_(const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(sv.data()))), 
          size_(sv.size()), owns_(false) {}

    // 4. 从 const std::string& 拷贝（持有模式）
    ByteBuffer(const std::string& s)
        : data_(new uint8_t[s.size()]), size_(s.size()), owns_(true) {
        std::memcpy(data_, s.data(), size_);
    }

    // 5. 从 std::string&& 移动（持有模式，将内容移入，原字符串置空）
    ByteBuffer(std::string&& s)
        : data_(new uint8_t[s.size()]), size_(s.size()), owns_(true) {
        std::memcpy(data_, s.data(), size_);
    }

    // 移动语义（转移所有权）
    ByteBuffer(ByteBuffer&& other) noexcept 
        : data_(std::exchange(other.data_, nullptr)),
          size_(std::exchange(other.size_, 0)),
          owns_(std::exchange(other.owns_, false)) {}

    // 移动赋值
    ByteBuffer& operator=(ByteBuffer&& other) noexcept {
        if (this != &other) {
            if (owns_) delete[] data_;
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0);
            owns_ = std::exchange(other.owns_, false);
        }
        return *this;
    }

    // 从 std::string&& 直接赋值（等价于先构造 ByteBuffer 再移动赋值，但更高效）
    ByteBuffer& operator=(std::string&& s) {
        if (owns_) delete[] data_;
        size_ = s.size();
        data_ = new uint8_t[size_];
        std::memcpy(data_, s.data(), size_);
        owns_ = true;
        return *this;
    }

    ~ByteBuffer() {
        if (owns_) {
            delete[] data_;
        }
    }

    // 禁止拷贝（防止浅拷贝导致双重释放）
    ByteBuffer(const ByteBuffer&) = delete;
    ByteBuffer& operator=(const ByteBuffer&) = delete;

    // 核心转换：转换为 C 字符串（不保证空终止，由调用方确保）
    const char* c_str() const noexcept {
        return reinterpret_cast<const char*>(data_);
    }

    // 核心转换：转换为 string_view
    std::string_view as_string_view() const noexcept {
        return std::string_view(reinterpret_cast<const char*>(data_), size_);
    }

    // 随机访问（Debug 模式下断言边界检查）
    uint8_t& operator[](size_t index) noexcept {
        assert(index < size_);
        return data_[index];
    }
    const uint8_t& operator[](size_t index) const noexcept {
        assert(index < size_);
        return data_[index];
    }

    // 迭代器（原生指针即满足 RandomAccessIterator）
    using iterator = uint8_t*;
    using const_iterator = const uint8_t*;

    iterator begin() noexcept { return data_; }
    iterator end() noexcept { return data_ + size_; }
    const_iterator begin() const noexcept { return data_; }
    const_iterator end() const noexcept { return data_ + size_; }
    const_iterator cbegin() const noexcept { return data_; }
    const_iterator cend() const noexcept { return data_ + size_; }

    // 访问原始数据
    uint8_t* data() noexcept { return data_; }
    const uint8_t* data() const noexcept { return data_; }
    size_t size() const noexcept { return size_; }

    // 提供类似 span 的操作
    std::span<uint8_t> as_span() noexcept { return std::span(data_, size_); }
    std::span<const uint8_t> as_span() const noexcept { return std::span(data_, size_); }

private:
    uint8_t* data_;
    size_t size_;
    bool owns_; // true 表示析构时需 delete[]
};

#endif // _BYTE_BUFFER_HPP_