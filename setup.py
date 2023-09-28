from setuptools import setup

package_name = 'planning_pkg'
lib_path = "planning_pkg/lib"

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, lib_path],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yongki',
    maintainer_email='kimyongwon987@naver.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "tryout_planning = planning_pkg.tryout_planning:main",
            "final_planning = planning_pkg.final_planning:main"
        ],
    },
)
