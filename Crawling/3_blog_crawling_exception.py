from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import csv
import time
import pandas as pd

def crawl_blog_post(post_url, driver):
    """
    네이버 블로그 게시글 URL에서 본문 내용과 이미지 URL을 추출하는 함수.
    '본문 내용 없음'이 발생할 경우, 상태나 예외 정보를 함께 반환하여
    어떤 이유로 크롤링이 실패했는지 확인 가능하도록 함.
    """
    try:
        driver.get(post_url)
        
        # 본문 컨테이너가 로딩될 때까지 대기
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.se-main-container"))
        )
        
        soup = BeautifulSoup(driver.page_source, "html.parser")
        content_div = soup.select_one("div.se-main-container")

        if not content_div:
            # 컨테이너 자체가 없으면 "EMPTY_CONTENT" 상태
            return "본문 내용 없음", [], "EMPTY_CONTENT"
        
        # 본문 추출
        content = content_div.get_text(separator='\n', strip=True)
        if not content.strip():
            # 컨테이너는 있는데 텍스트가 비어있는 경우
            return "본문 내용 없음", [], "EMPTY_CONTENT"
        
        # 이미지 URL 추출
        img_tags = soup.select("div.se-main-container img")
        img_urls = [
            (img_tag.get("data-lazy-src") or img_tag.get("src"))
            for img_tag in img_tags
            if (img_tag.get("data-lazy-src") or img_tag.get("src"))
        ]
        
        # 정상적으로 본문이 있는 경우
        return content, img_urls, "SUCCESS"

    except Exception as e:
        # 네트워크 문제, 타임아웃, 기타 예외
        print(f"[오류] 페이지 요청 실패: {post_url}\n오류 메시지: {e}")
        return None, None, f"ERROR: {str(e)}"


def crawl_naver_blog(blog_id, category_no, driver, output_file="blog_posts.csv"):
    """
    네이버 블로그 특정 카테고리에서 모든 게시글을 크롤링하고 
    본문 내용과 이미지 URL, 상태 등을 CSV 파일로 저장하는 함수
    """
    base_url = "https://blog.naver.com/PostList.naver"
    
    # CSV에 'Status' 열을 추가해 크롤링 상태 기록
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['Title', 'Link', 'Content', 'Image URLs', 'Status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        page = 1
        while True:
            print(f"페이지 {page} 크롤링 시작...")
            params = f"?blogId={blog_id}&categoryNo={category_no}&currentPage={page}&from=postList"
            url = base_url + params

            driver.get(url)
            try:
                # 게시글 목록 로딩 대기
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "tbody#postBottomTitleListBody > tr"))
                )
            except:
                print(f"페이지 {page}에서 더 이상 게시글을 찾지 못했습니다. 종료합니다.")
                break

            soup = BeautifulSoup(driver.page_source, "html.parser")
            rows = soup.select("tbody#postBottomTitleListBody > tr")
            
            if not rows:
                print(f"페이지 {page}에 게시글이 없습니다. 종료합니다.")
                break

            for row in rows:
                a_tag = row.select_one("td.title div.wrap_td span.ell2 a.pcol2")
                if not a_tag:
                    print("a 태그를 찾지 못했습니다. (게시글 링크 없음)")
                    continue

                relative_url = a_tag.get("href", "")
                link = "https://blog.naver.com" + relative_url
                title = a_tag.get_text(strip=True)

                print(f"  게시글 '{title}' 크롤링 중...")
                content, img_urls, status = crawl_blog_post(link, driver)

                writer.writerow({
                    'Title': title,
                    'Link': link,
                    'Content': content if content else "",
                    'Image URLs': ', '.join(img_urls) if img_urls else "",
                    'Status': status
                })

            page += 1
            time.sleep(0.5)


if __name__ == "__main__":
    chrome_options = Options()
    #chrome_options.add_argument("--headless")  # 브라우저 창을 띄우지 않도록 설정 (원하면 주석 해제)
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    service = Service(r'C:\Users\r2com\Desktop\VSCode\chromedriver.exe')
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    # 블로그 목록 예시
    file_path = r'C:\Users\r2com\Desktop\VSCode\blog_list.xlsx'
    blog_list = pd.read_excel(file_path, dtype={'Category_Num_List': str})
    
    for index, blog in blog_list.iterrows():
        category_list = blog['Category_Num_List'].split(',')
        for idx, category_num in enumerate(category_list): # enumerate() 사용
            crawl_naver_blog(
                blog_id=blog['Blog_Name'],
                category_no=int(category_num),
                driver=driver,  # Ensure to pass the driver here
                output_file=f"{blog['Name']}_category_index_{idx}.csv" # 인덱스 사용
            ) 
    driver.quit()
