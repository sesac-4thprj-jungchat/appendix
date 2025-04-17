import requests
from bs4 import BeautifulSoup
import time
import csv
import os
import pandas as pd

def crawl_blog_post(post_url):
    try:
        resp = requests.get(post_url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        content_div = soup.select_one("div.se-main-container")
        content = content_div.get_text(separator='\n', strip=True) if content_div else "본문 내용 없음"

        img_tags = soup.select("div.se-main-container img")
        img_urls = [img_tag.get("data-lazy-src") or img_tag.get("src") for img_tag in img_tags if img_tag.get("data-lazy-src") or img_tag.get("src")]

        return content, img_urls

    except requests.exceptions.RequestException as e:
        print(f"페이지 요청 실패: {post_url}, 오류: {e}")
        return None, None

def crawl_naver_blog(blog_id, category_no, output_file="blog_posts.csv"):
    base_url = "https://blog.naver.com/PostList.naver"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            " AppleWebKit/537.36 (KHTML, like Gecko)"
            " Chrome/88.0.4324.150 Safari/537.36"
        )
    }

    with open(output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['Title', 'Link', 'Content', 'Image URLs']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        page = 1
        while True:
            print(f"페이지 {page} 크롤링 시작...")
            params = {
                "blogId": blog_id,
                "categoryNo": category_no,
                "currentPage": page,
                "from": "postList"
            }

            resp = requests.get(base_url, params=params, headers=headers)
            if resp.status_code != 200:
                print(f"[ERROR] Page {page} - status code: {resp.status_code}")
                break

            soup = BeautifulSoup(resp.text, "html.parser")

            li_elements = soup.select("#PostThumbnailAlbumViewArea > ul > li")

            if not li_elements:
                print(f"No more posts found at page {page}. Stopping.")
                break

            for li in li_elements:
                link_a = li.select_one("a")
                if not link_a:
                    continue

                link = "https://blog.naver.com" + link_a.get("href", "")

                title_strong = link_a.select_one("div.area_text > strong")
                title = title_strong.get_text(strip=True) if title_strong else "No Title"

                print(f"  게시글 '{title}' 크롤링...")
                content, img_urls = crawl_blog_post(link)

                writer.writerow({'Title': title, 'Link': link, 'Content': content, 'Image URLs': ', '.join(img_urls)})

            page += 1
            time.sleep(2)  # Increased delay to 2 seconds

if __name__ == "__main__":
    file_path = r'C:\Users\r2com\Desktop\VSCode\blog_list.xlsx'
    blog_list = pd.read_excel(file_path, dtype={'Category_Num_List': str})

    for index, blog in blog_list.iterrows():
        category_list = blog['Category_Num_List'].split(',')
        for idx, category_num in enumerate(category_list):
            crawl_naver_blog(
                blog_id=blog['Blog_Name'],
                category_no=int(category_num),
                output_file=f"{blog['Name']}_category_index_{idx}.csv"
            )
