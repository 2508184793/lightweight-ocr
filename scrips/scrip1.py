"""
PDF识别脚本
1. 连接数据库scan_files表
2. 获取字段path, scan_data_base_id
3. 识别path路径的PDF文件
4. 将识别的文本写入scan_text表
5. 更新scan_files表的status为1（成功）或3（失败）
6. 更新ScanDataBase表的is_syn为0
"""

import os
import re
import sys
import pymysql
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr_engine import OCREngine


def connect_database():
    """
    连接数据库
    """
    try:
        conn = pymysql.connect(
            host='192.168.10.12',  # 数据库地址
            port=3306,         # 端口
            user='root',       # 用户名
            password='XU3,q.a3]61Ppebbf~4%',  # 密码
            database='scan_system',  # 数据库名
            charset='utf8mb4'
        )
        print("数据库连接成功")
        return conn
    except Exception as e:
        print(f"数据库连接失败: {e}")
        return None


def get_pdf_files(conn):
    """
    从scan_files表获取一个PDF文件路径和scan_data_base_id
    """
    try:
        cursor = conn.cursor()
        # 只获取一个未处理的PDF文件 (status=0)
        sql = """
        SELECT path, scan_data_base_id 
        FROM scan_files 
        WHERE path LIKE '%.pdf' AND (status = 0 OR status IS NULL)
        LIMIT 1
        """
        cursor.execute(sql)
        result = cursor.fetchone()
        cursor.close()
        if result:
            print(f"找到1个PDF文件: {result[0]}")
            return [result]
        else:
            print("未找到PDF文件")
            return []
    except Exception as e:
        print(f"查询失败: {e}")
        return []


def recognize_pdf(pdf_path):
    """
    识别PDF文件
    返回: (text, success) 元组
    """
    try:
        print(f"开始识别PDF: {pdf_path}")
        engine = OCREngine(lang='ch')
        result = engine.recognize_pdf(pdf_path, dpi=150)
        print(f"识别完成，总页数: {result.total_pages}")
        return result.text, True
    except Exception as e:
        print(f"识别失败: {e}")
        return "", False


def insert_scan_text(conn, site_text, scan_data_base_id):
    """
    将识别结果写入scan_text表
    """
    try:
        # 转换scan_data_base_id为bigint
        try:
            scan_data_base_id_bigint = int(scan_data_base_id)
        except:
            # 提取数字部分
            scan_data_base_id_bigint = int(re.search(r'\d+', scan_data_base_id).group())
        
        cursor = conn.cursor()
        sql = """
        INSERT INTO scan_text (site_text, scan_data_base_id)
        VALUES (%s, %s)
        """
        cursor.execute(sql, (site_text, scan_data_base_id_bigint))
        conn.commit()
        cursor.close()
        print(f"数据写入成功, scan_data_base_id: {scan_data_base_id_bigint}")
        return True
    except Exception as e:
        print(f"写入失败: {e}")
        return False


def update_scan_files_status(conn, path, status=1):
    """
    更新scan_files表的status
    status: 1=成功, 3=失败
    """
    try:
        cursor = conn.cursor()
        sql = """
        UPDATE scan_files 
        SET status = %s 
        WHERE path = %s
        """
        cursor.execute(sql, (status, path))
        conn.commit()
        cursor.close()
        status_text = "成功" if status == 1 else "失败"
        print(f"scan_files表status更新为{status}({status_text}): {path}")
        return True
    except Exception as e:
        print(f"status更新失败: {e}")
        return False


def update_scan_database_is_syn(conn, scan_data_base_id):
    """
    更新ScanDataBase表的is_syn为0
    """
    try:
        # 转换scan_data_base_id为bigint
        try:
            scan_data_base_id_bigint = int(scan_data_base_id)
        except:
            # 提取数字部分
            scan_data_base_id_bigint = int(re.search(r'\d+', scan_data_base_id).group())
        
        cursor = conn.cursor()
        sql = """
        UPDATE ScanDataBase 
        SET is_syn = 0 
        WHERE id = %s
        """
        cursor.execute(sql, (scan_data_base_id_bigint,))
        conn.commit()
        cursor.close()
        print(f"ScanDataBase表is_syn更新成功: {scan_data_base_id_bigint}")
        return True
    except Exception as e:
        print(f"is_syn更新失败: {e}")
        return False


def main():
    """
    主函数 - 有PDF时立即处理，没有时等待20秒
    """
    import time
    
    # 连接数据库
    conn = connect_database()
    if not conn:
        return
    
    try:
        print("开始循环检测...")
        
        while True:
            # 获取一个PDF文件
            pdf_files = get_pdf_files(conn)
            
            if not pdf_files:
                # 没有PDF文件，等待20秒后再次检测
                print("没有待处理的PDF文件，等待20秒后再次检测...")
                time.sleep(20)
                continue
            
            # 有PDF文件，立即处理
            for path, scan_data_base_id in pdf_files:
                # 检查文件是否存在
                pdf_path = Path(path)
                if not pdf_path.exists():
                    print(f"文件不存在: {path}")
                    # 文件不存在，status改为3（失败）
                    update_scan_files_status(conn, path, status=3)
                    # 更新ScanDataBase表的is_syn为0
                    update_scan_database_is_syn(conn, scan_data_base_id)
                    continue
                
                # 识别PDF
                text, recognize_success = recognize_pdf(str(pdf_path))
                
                if recognize_success and text:
                    # 识别成功且有内容，写入数据库
                    success = insert_scan_text(conn, text, scan_data_base_id)
                    if success:
                        # 写入成功，status改为1
                        update_scan_files_status(conn, path, status=1)
                        # 更新ScanDataBase表的is_syn为0
                        update_scan_database_is_syn(conn, scan_data_base_id)
                    else:
                        # 写入失败，status改为3
                        update_scan_files_status(conn, path, status=3)
                        # 更新ScanDataBase表的is_syn为0
                        update_scan_database_is_syn(conn, scan_data_base_id)
                elif recognize_success and not text:
                    # 识别成功但内容为空，不写入scan_text，status改为1
                    print(f"识别成功但内容为空: {path}")
                    update_scan_files_status(conn, path, status=1)
                    # 更新ScanDataBase表的is_syn为0
                    update_scan_database_is_syn(conn, scan_data_base_id)
                else:
                    # 识别失败，不写入scan_text，status改为3
                    print(f"识别失败: {path}")
                    update_scan_files_status(conn, path, status=3)
                    # 更新ScanDataBase表的is_syn为0
                    update_scan_database_is_syn(conn, scan_data_base_id)
            
            # 处理完一个PDF后，立即继续检测下一个（不等待）
            print("继续检测下一个PDF...")
                
    except KeyboardInterrupt:
        print("程序被手动中断")
    finally:
        if conn:
            conn.close()
            print("数据库连接已关闭")


if __name__ == "__main__":
    main()
