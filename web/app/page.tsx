import axios from "axios";
import ClientPage from "./client-page";

// 서버 컴포넌트에서 데이터 가져오기
async function getStockData() {
  try {
    const response = await axios.get(
      `${process.env.NEXT_PUBLIC_API_URL}/predict/%5EIXIC`
    );
    return response.data;
  } catch (error) {
    console.error("Failed to fetch stock data:", error);
    return undefined; // null 대신 undefined 반환
  }
}

export default async function Home() {
  // 서버에서 데이터 가져오기
  const stockData = await getStockData();

  return <ClientPage initialData={stockData} />;
}
