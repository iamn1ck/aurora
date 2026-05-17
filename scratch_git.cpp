#include <iostream>
#include <cstdlib>

int main() {
    std::cout << "--- SEARCHING FOR WrapVulkanImage IN GIT LOG ---" << std::endl;
    std::system("git log --all -S WrapVulkanImage --oneline");

    std::cout << "\n--- SEARCHING FOR ExternalImageDescriptorOpaqueFD IN GIT LOG ---" << std::endl;
    std::system("git log --all -S ExternalImageDescriptorOpaqueFD --oneline");

    std::cout << "\n--- SEARCHING FOR ExternalImageExportInfoOpaqueFD IN GIT LOG ---" << std::endl;
    std::system("git log --all -S ExternalImageExportInfoOpaqueFD --oneline");

    std::cout << "\n--- SEARCHING FOR get_vulkan_graphics_device IN GIT LOG ---" << std::endl;
    std::system("git log --all -S get_vulkan_graphics_device --oneline");

    return 0;
}
