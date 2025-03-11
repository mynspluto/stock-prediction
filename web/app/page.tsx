import axios from "axios";
import ClientPage from "./client-page";

// 서버 컴포넌트에서 데이터 가져오기
async function getStockData() {
  // 현재 시간과 오전 8시 비교하여 캐싱 제어
  const now = new Date();
  const hour = now.getHours();
  const target = 8; // 오전 8시

  // 오전 8시에 항상 새로운 데이터를 가져오기 위한 파라미터
  const cacheParam = `${now.getFullYear()}-${now.getMonth()}-${now.getDate()}-${
    hour >= target ? 1 : 0
  }`;

  try {
    const response = await axios.get(
      `${process.env.NEXT_PUBLIC_API_URL}/predict/%5EIXIC?cache=${cacheParam}`
    );
    return response.data;
  } catch (error) {
    console.error("Failed to fetch stock data:", error);
    return undefined;
  }
}

export default async function Home() {
  const stockData = await getStockData();
  return <ClientPage initialData={stockData} />;
}
